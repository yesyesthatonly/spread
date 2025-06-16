import unittest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import os
import sys
import json
import grpc

# Import Hugging Face and PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
import torch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from orchestrator import orchestrator as orch
from protos import tensor_parallel_pb2
from protos import tensor_parallel_pb2_grpc

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.test_model_name = "gpt2" # Use a small standard model for config
        self.test_config_path = os.path.join(project_root, "tests", "test_orchestrator_config.json")
        self.shard_configs_data = [
            {"shard_id": "shard0", "ip": "localhost", "port": 50070, "slice_start": 0, "slice_end": 10, "axis": "columns", "model_name": self.test_model_name},
            {"shard_id": "shard1", "ip": "localhost", "port": 50071, "slice_start": 10, "slice_end": 20, "axis": "columns", "model_name": self.test_model_name}
        ]
        with open(self.test_config_path, 'w') as f:
            json.dump(self.shard_configs_data, f)

        # Mock Hugging Face model and tokenizer loading
        self.mock_orchestrator_hf_model = MagicMock(spec=AutoModelForCausalLM)
        self.mock_orchestrator_tokenizer = MagicMock(spec=AutoTokenizer)

        # Configure mock model attributes accessed in Orchestrator
        self.model_config = GPT2Config(n_layer=2, n_embd=768, vocab_size=50257) # Standard GPT-2 hidden size
        self.mock_orchestrator_hf_model.config = self.model_config
        self.mock_orchestrator_hf_model.transformer.h = [MagicMock(spec=torch.nn.Module) for _ in range(self.model_config.n_layer)]
        for layer_mock in self.mock_orchestrator_hf_model.transformer.h:
            layer_mock.ln_1 = MagicMock(spec=torch.nn.LayerNorm)
            layer_mock.attn = MagicMock(spec=torch.nn.Module) # Mocking the whole attention block for now
            layer_mock.ln_2 = MagicMock(spec=torch.nn.LayerNorm)
            layer_mock.mlp = MagicMock(spec=torch.nn.Module) # Mocking MLP, its sub-parts will be called
            layer_mock.mlp.c_fc = MagicMock(spec=torch.nn.Linear)
            layer_mock.mlp.c_fc.bias = MagicMock(data=torch.randn(4 * self.model_config.n_embd))
            layer_mock.mlp.c_proj = MagicMock(spec=torch.nn.Linear)
            layer_mock.mlp.c_proj.bias = MagicMock(data=torch.randn(self.model_config.n_embd))


        self.mock_orchestrator_hf_model.transformer.wte = MagicMock(spec=torch.nn.Embedding)
        self.mock_orchestrator_hf_model.transformer.wpe = MagicMock(spec=torch.nn.Embedding)
        self.mock_orchestrator_hf_model.transformer.drop = MagicMock(spec=torch.nn.Dropout)
        self.mock_orchestrator_hf_model.transformer.ln_f = MagicMock(spec=torch.nn.LayerNorm)
        self.mock_orchestrator_hf_model.lm_head = MagicMock(spec=torch.nn.Linear)


        # Configure mock tokenizer attributes
        self.mock_orchestrator_tokenizer.pad_token = None # Simulate it being initially None
        self.mock_orchestrator_tokenizer.eos_token = "<|endoftext|>"
        self.mock_orchestrator_tokenizer.eos_token_id = 50256


        self.automodel_patcher = patch('orchestrator.orchestrator.AutoModelForCausalLM.from_pretrained', return_value=self.mock_orchestrator_hf_model)
        self.autotokenizer_patcher = patch('orchestrator.orchestrator.AutoTokenizer.from_pretrained', return_value=self.mock_orchestrator_tokenizer)

        self.mock_automodel_from_pretrained = self.automodel_patcher.start()
        self.mock_autotokenizer_from_pretrained = self.autotokenizer_patcher.start()

        # Patch grpc.insecure_channel and the stub (for run_inference_layer_matmul)
        self.mock_channel = MagicMock()
        self.mock_stub = MagicMock(spec=tensor_parallel_pb2_grpc.TensorParallelServiceStub)
        self.grpc_insecure_channel_patcher = patch('grpc.insecure_channel', return_value=self.mock_channel)
        self.tensor_parallel_service_stub_patcher = patch('protos.tensor_parallel_pb2_grpc.TensorParallelServiceStub', return_value=self.mock_stub)
        self.mock_grpc_insecure_channel = self.grpc_insecure_channel_patcher.start()
        self.mock_tensor_parallel_service_stub = self.tensor_parallel_service_stub_patcher.start()

        # Instantiate Orchestrator
        self.orchestrator_instance = orch.Orchestrator(config_path=self.test_config_path, model_name=self.test_model_name)

    def tearDown(self):
        self.automodel_patcher.stop()
        self.autotokenizer_patcher.stop()
        self.grpc_insecure_channel_patcher.stop()
        self.tensor_parallel_service_stub_patcher.stop()
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_orchestrator_initialization_loads_model_tokenizer(self):
        self.mock_automodel_from_pretrained.assert_called_once_with(self.test_model_name)
        self.mock_autotokenizer_from_pretrained.assert_called_once_with(self.test_model_name)

        self.assertIsNotNone(self.orchestrator_instance.model)
        self.assertIsNotNone(self.orchestrator_instance.tokenizer)

        self.mock_orchestrator_hf_model.to.assert_called_once_with(self.orchestrator_instance.device)
        self.mock_orchestrator_hf_model.eval.assert_called_once()

        # Check if pad_token was set correctly
        self.assertEqual(self.mock_orchestrator_tokenizer.pad_token, self.mock_orchestrator_tokenizer.eos_token)

    def test_load_config(self): # Existing test, should still pass
        self.assertEqual(len(self.orchestrator_instance.shard_configs), 2)
        self.assertEqual(self.orchestrator_instance.shard_configs[0]['shard_id'], 'shard0')

    def test_connect_to_shards(self): # Existing test, should still pass
        self.mock_grpc_insecure_channel.assert_any_call(f"{self.shard_configs_data[0]['ip']}:{self.shard_configs_data[0]['port']}")
        self.mock_grpc_insecure_channel.assert_any_call(f"{self.shard_configs_data[1]['ip']}:{self.shard_configs_data[1]['port']}")
        self.assertEqual(self.mock_grpc_insecure_channel.call_count, 2)
        self.mock_tensor_parallel_service_stub.assert_any_call(self.mock_channel)
        self.assertEqual(self.mock_tensor_parallel_service_stub.call_count, 2)

    def test_prepare_input_tensor_bytes(self): # Renamed and updated
        numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = self.orchestrator_instance._prepare_input_tensor(numpy_array)
        self.assertEqual(list(tensor_proto.dims), [2, 2])
        self.assertEqual(tensor_proto.serialized_data, numpy_array.tobytes())
        self.assertEqual(tensor_proto.dtype, str(numpy_array.dtype))

    def test_process_output_tensor_bytes(self): # Renamed and updated
        expected_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = tensor_parallel_pb2.Tensor(
            dims=[2,2],
            serialized_data=expected_array.tobytes(),
            dtype=str(expected_array.dtype)
        )
        numpy_array = self.orchestrator_instance._process_output_tensor(tensor_proto)
        np.testing.assert_array_equal(numpy_array, expected_array)

    def test_run_inference_layer_matmul_success_sends_tensor_name(self): # Updated
        input_np = np.array([[1, 2, 3, 4]], dtype=np.float32)
        test_tensor_name = "transformer.h.0.mlp.c_fc.weight" # Example tensor name

        output_slice0_np = np.array([[10, 11]], dtype=np.float32)
        output_slice1_np = np.array([[12, 13]], dtype=np.float32)

        # Ensure the output tensor protos use the new byte serialization format
        response0_proto = tensor_parallel_pb2.ComputeResponse(
            output_tensor=tensor_parallel_pb2.Tensor(dims=list(output_slice0_np.shape), serialized_data=output_slice0_np.tobytes(), dtype=str(output_slice0_np.dtype))
        )
        response1_proto = tensor_parallel_pb2.ComputeResponse(
            output_tensor=tensor_parallel_pb2.Tensor(dims=list(output_slice1_np.shape), serialized_data=output_slice1_np.tobytes(), dtype=str(output_slice1_np.dtype))
        )
        self.mock_stub.ComputeMatMul.side_effect = [response0_proto, response1_proto]

        result = self.orchestrator_instance.run_inference_layer_matmul(
            input_np,
            layer_id="test_layer",
            tensor_name_for_shards=test_tensor_name
        )

        self.assertIsNotNone(result)
        expected_result = np.concatenate((output_slice0_np, output_slice1_np), axis=1)
        np.testing.assert_array_almost_equal(result, expected_result)

        self.assertEqual(self.mock_stub.ComputeMatMul.call_count, 2)
        # Check that tensor_name was passed correctly in requests
        calls = self.mock_stub.ComputeMatMul.call_args_list
        for c in calls:
            request_sent = c[0][0] # First arg of first call
            self.assertEqual(request_sent.tensor_name, test_tensor_name)
            self.assertEqual(request_sent.model_name, self.test_model_name)

    @patch.object(orch.Orchestrator, 'run_inference_layer_matmul')
    def test_full_inference_basic_flow(self, mock_run_sharded_matmul):
        # --- Setup Mocks for full_inference ---
        test_input_text = "Hello world"
        max_gen_length = 3 # input_len + 2 new tokens

        # Mock tokenizer
        # input_ids for "Hello world" (example, actual IDs depend on tokenizer)
        initial_input_ids = torch.tensor([[15496, 995]], device=self.orchestrator_instance.device) # (batch, seq)
        self.mock_orchestrator_tokenizer.return_value = {
            "input_ids": initial_input_ids,
            "attention_mask": torch.ones_like(initial_input_ids)
        }
        self.mock_orchestrator_tokenizer.eos_token_id = 50256 # Standard for GPT-2
        self.mock_orchestrator_tokenizer.decode.return_value = "mocked output text"

        # Mock model components (embeddings, layer outputs, lm_head)
        hidden_size = self.model_config.n_embd # 768

        # Embeddings
        self.mock_orchestrator_hf_model.transformer.wte.return_value = torch.randn(1, initial_input_ids.shape[1], hidden_size, device=self.orchestrator_instance.device)
        self.mock_orchestrator_hf_model.transformer.wpe.return_value = torch.randn(1, initial_input_ids.shape[1], hidden_size, device=self.orchestrator_instance.device)
        self.mock_orchestrator_hf_model.transformer.drop.side_effect = lambda x: x # Identity function for dropout

        # Transformer block layer outputs (mocked to just pass data through or return new random)
        for i, block_mock in enumerate(self.mock_orchestrator_hf_model.transformer.h):
            # ln_1 and ln_2
            block_mock.ln_1.side_effect = lambda x: x + 0.1 # Simulate some change
            block_mock.ln_2.side_effect = lambda x: x + 0.2 # Simulate some change
            # attn (non-sharded part) - returns tuple (hidden_states, present_key_value, optional_attentions)
            block_mock.attn.return_value = (torch.randn(1, initial_input_ids.shape[1] + i , hidden_size, device=self.orchestrator_instance.device),) # +i for growing seq len

        # Mock run_inference_layer_matmul (sharded MLP layers)
        # It's called for c_fc and c_proj for each layer.
        # Needs to return shapes compatible with what full_inference expects.
        # Output of c_fc: (batch * seq, 4 * hidden_size)
        # Output of c_proj: (batch * seq, hidden_size)
        def matmul_side_effect(input_np, layer_id, tensor_name_for_shards):
            batch_seq_combined, features_in = input_np.shape
            if "mlp.c_fc.weight" in tensor_name_for_shards:
                # Simulate output that would be (batch*seq, intermediate_mlp_dim)
                # where intermediate_mlp_dim is sum of shard slice_widths for c_fc
                # For simplicity, assume it's 4 * hidden_size (GPT-2 standard)
                return np.random.randn(batch_seq_combined, 4 * hidden_size).astype(np.float32)
            elif "mlp.c_proj.weight" in tensor_name_for_shards:
                # Simulate output (batch*seq, hidden_size)
                return np.random.randn(batch_seq_combined, hidden_size).astype(np.float32)
            return None
        mock_run_sharded_matmul.side_effect = matmul_side_effect

        # Final layer norm
        self.mock_orchestrator_hf_model.transformer.ln_f.side_effect = lambda x: x + 0.3

        # LM Head - for each step of generation, it will predict next token
        # It receives hidden_states[:, -1, :] which is (batch, hidden_size)
        # It should output (batch, vocab_size)
        # For testing, make it deterministically pick a new token ID
        next_token_ids_to_generate = [torch.tensor([[100]], device=self.orchestrator_instance.device),
                                      torch.tensor([[self.mock_orchestrator_tokenizer.eos_token_id]], device=self.orchestrator_instance.device)]
        self.mock_orchestrator_hf_model.lm_head.side_effect = [
            torch.nn.functional.one_hot(next_token_ids_to_generate[0][0], num_classes=self.model_config.vocab_size).float(),
            torch.nn.functional.one_hot(next_token_ids_to_generate[1][0], num_classes=self.model_config.vocab_size).float(),
        ]

        # --- Execute full_inference ---
        result_text = self.orchestrator_instance.full_inference(test_input_text, max_length=max_gen_length - initial_input_ids.shape[1])

        # --- Assertions ---
        self.mock_orchestrator_tokenizer.assert_called_once_with(test_input_text, return_tensors="pt", padding=True)

        # Expected calls to sharded matmul: n_layers * 2 (c_fc, c_proj per layer) * generation_steps
        # Generation steps = max_length - initial_input_ids.shape[1] + 1 (for the initial prompt processing)
        # Here, generation_steps = (3 - 2) + 1 = 2 (one for prompt, one for 1st new token, then EOS)
        num_generation_steps = (max_gen_length - initial_input_ids.shape[1]) + 1
        expected_matmul_calls = self.model_config.n_layer * 2 * num_generation_steps
        self.assertEqual(mock_run_sharded_matmul.call_count, expected_matmul_calls)

        # Check tensor_names passed to sharded matmul
        for i in range(self.model_config.n_layer):
            mock_run_sharded_matmul.assert_any_call(ANY, layer_id=f"layer_{i}_mlp_fc", tensor_name_for_shards=f"transformer.h.{i}.mlp.c_fc.weight")
            mock_run_sharded_matmul.assert_any_call(ANY, layer_id=f"layer_{i}_mlp_proj", tensor_name_for_shards=f"transformer.h.{i}.mlp.c_proj.weight")

        self.mock_orchestrator_hf_model.lm_head.assert_called()
        self.assertEqual(self.mock_orchestrator_hf_model.lm_head.call_count, num_generation_steps)

        self.mock_orchestrator_tokenizer.decode.assert_called_once()
        self.assertEqual(result_text, "mocked output text")


if __name__ == "__main__":
    unittest.main()
