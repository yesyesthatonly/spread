import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys
import json
import grpc

# Import Hugging Face and PyTorch
from transformers import AutoConfig, GPT2Model # Using GPT2Model as a spec for mocking
import torch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from protos import tensor_parallel_pb2
from shard import shard # Assuming shard.py is in shard/ directory

class TestShard(unittest.TestCase):

    def setUp(self):
        # Create a dummy config file for testing
        self.test_config_path = os.path.join(project_root, "tests", "test_shard_config.json")

        # Shard configuration for tests - using "gpt2" (a smaller model) for faster test setup
        # if the mock actually tried to download, though from_pretrained is mocked.
        self.shard0_config = {
            "shard_id": "test_shard0", "ip": "localhost", "port": 50090,
            "slice_start": 0, "slice_end": 10, "axis": "columns", "model_name": "gpt2"
        }
        self.shard1_config_rows = { # This config is for a hypothetical row-sharding test
            "shard_id": "test_shard1_rows", "ip": "localhost", "port": 50091,
            "slice_start": 0, "slice_end": 5, "axis": "rows", "model_name": "gpt2"
        }
        with open(self.test_config_path, 'w') as f:
            json.dump([self.shard0_config, self.shard1_config_rows], f)

        # Mock Hugging Face model loading (AutoModelForCausalLM.from_pretrained)
        # We use a spec (GPT2Model) to make the mock behave more like the actual object.
        self.mock_hf_model = MagicMock(spec=GPT2Model)

        # Configure the mock model with a config object similar to a real model's config
        # Using "gpt2" config as a base. n_embd is the hidden size.
        mock_model_config = AutoConfig.from_pretrained("gpt2")
        mock_model_config.n_layer = 2 # Reduce number of layers for simplicity in tests
        self.mock_hf_model.config = mock_model_config

        # Create dummy tensors for the mock model's parameters.
        # These names must match the templates in shard.py's _load_all_tensor_slices
        # For GPT-2, n_embd (hidden size) is 768.
        # c_fc.weight shape is typically (n_embd, 4 * n_embd)
        # c_proj.weight shape is typically (4 * n_embd, n_embd)
        # c_attn.weight shape is typically (n_embd, 3 * n_embd) for QKV combined

        self.param_mlp_fc_name_l0 = "transformer.h.0.mlp.c_fc.weight"
        self.param_mlp_fc_data_l0 = torch.randn(mock_model_config.n_embd, mock_model_config.n_embd * 4)

        self.param_attn_qkv_name_l1 = "transformer.h.1.attn.c_attn.weight"
        self.param_attn_qkv_data_l1 = torch.randn(mock_model_config.n_embd, mock_model_config.n_embd * 3)

        # The shard's _load_all_tensor_slices uses `param.data`
        # So, each "parameter" in named_parameters should be a MagicMock that has a .data attribute.
        mock_param_mlp_l0 = MagicMock()
        mock_param_mlp_l0.data = self.param_mlp_fc_data_l0

        mock_param_attn_l1 = MagicMock()
        mock_param_attn_l1.data = self.param_attn_qkv_data_l1

        self.mock_hf_model.named_parameters.return_value = [
            (self.param_mlp_fc_name_l0, mock_param_mlp_l0),
            (self.param_attn_qkv_name_l1, mock_param_attn_l1),
        ]

        # Patch 'AutoModelForCausalLM.from_pretrained' in the shard.shard module
        self.automodel_patcher = patch('shard.shard.AutoModelForCausalLM.from_pretrained', return_value=self.mock_hf_model)
        self.mock_automodel_from_pretrained = self.automodel_patcher.start()

        # Patch environment variables for shard initialization.
        # This needs to be done *before* TensorParallelServiceServicer is instantiated.
        self.env_patch = patch.dict(os.environ, {
            "SHARD_ID": "test_shard0",
            "CONFIG_PATH": self.test_config_path
        })
        self.env_patch.start()

        # Instantiate the servicer. This will trigger _initialize_shard,
        # which includes model loading (mocked) and tensor slice caching.
        self.servicer = shard.TensorParallelServiceServicer()

    def tearDown(self):
        self.automodel_patcher.stop()
        self.env_patch.stop() # Stop the environment patch
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_initialize_shard_loads_slices(self):
        # _initialize_shard is called during self.servicer instantiation in setUp.

        # Verify that from_pretrained was called with the model_name from shard0_config
        self.mock_automodel_from_pretrained.assert_called_once_with("gpt2")

        # Check if the expected tensor slice (param_mlp_fc_name_l0) is in the cache
        self.assertIn(self.param_mlp_fc_name_l0, self.servicer.tensor_slices_cache)
        cached_slice_mlp_l0 = self.servicer.tensor_slices_cache[self.param_mlp_fc_name_l0]

        # Determine expected shape: (n_embd, slice_end - slice_start)
        n_embd = self.mock_hf_model.config.n_embd # 768 for gpt2
        expected_cols = self.shard0_config["slice_end"] - self.shard0_config["slice_start"] # 10
        self.assertEqual(cached_slice_mlp_l0.shape, (n_embd, expected_cols))

        # Verify the content of the slice
        expected_slice_data_torch = self.param_mlp_fc_data_l0[:, self.shard0_config["slice_start"]:self.shard0_config["slice_end"]]
        np.testing.assert_array_almost_equal(cached_slice_mlp_l0, expected_slice_data_torch.cpu().numpy())

        # Check the other tensor defined (param_attn_qkv_name_l1 for layer 1)
        # Since _load_all_tensor_slices iterates num_layers (mocked to 2), this should also be loaded.
        self.assertIn(self.param_attn_qkv_name_l1, self.servicer.tensor_slices_cache)
        cached_slice_attn_l1 = self.servicer.tensor_slices_cache[self.param_attn_qkv_name_l1]
        self.assertEqual(cached_slice_attn_l1.shape, (n_embd, expected_cols)) # Also column sharded per test_shard0 config
        expected_slice_attn_data_torch = self.param_attn_qkv_data_l1[:, self.shard0_config["slice_start"]:self.shard0_config["slice_end"]]
        np.testing.assert_array_almost_equal(cached_slice_attn_l1, expected_slice_attn_data_torch.cpu().numpy())


    def test_compute_matmul_real_slice(self):
        # This test uses a slice loaded during setUp (param_mlp_fc_name_l0)
        test_tensor_name = self.param_mlp_fc_name_l0

        # The slice is (K, N_slice) = (n_embd, 10) = (768, 10) as per shard0_config and gpt2 base
        slice_data_np = self.servicer.tensor_slices_cache[test_tensor_name]

        # Input tensor (B, K) = (3, n_embd) = (3, 768)
        input_data_np = np.full((3, self.mock_hf_model.config.n_embd), 2.0, dtype=np.float32)

        input_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(input_data_np.shape),
            serialized_data=input_data_np.tobytes(), # New serialization
            dtype=str(input_data_np.dtype)
        )
        request = tensor_parallel_pb2.ComputeRequest(
            input_tensor=input_tensor_proto,
            tensor_name=test_tensor_name # Specify the tensor name for the shard to use
        )

        mock_context = MagicMock()
        response = self.servicer.ComputeMatMul(request, mock_context)

        mock_context.abort.assert_not_called()
        self.assertIsNotNone(response.output_tensor)

        output_dims = list(response.output_tensor.dims)
        output_dtype = np.dtype(response.output_tensor.dtype)
        output_data = np.frombuffer(response.output_tensor.serialized_data, dtype=output_dtype).reshape(output_dims)

        # Expected shape: (Batch_size, N_slice) = (3, 10)
        self.assertEqual(output_data.shape, (input_data_np.shape[0], slice_data_np.shape[1]))
        self.assertEqual(output_data.dtype, np.float32)

        # Verify actual computation
        expected_output_np = np.matmul(input_data_np, slice_data_np)
        np.testing.assert_array_almost_equal(output_data, expected_output_np, decimal=5)


    def test_compute_matmul_value_error_real_slice(self):
        test_tensor_name = "test_value_error_tensor" # A specific name for this test
        # K=5 for the slice
        self.servicer.tensor_slices_cache[test_tensor_name] = np.ones((5, 5), dtype=np.float32)

        # Input data with K=4 (incompatible for matmul)
        input_data_np = np.ones((3, 4), dtype=np.float32)

        input_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(input_data_np.shape),
            serialized_data=input_data_np.tobytes(),
            dtype=str(input_data_np.dtype)
        )
        request = tensor_parallel_pb2.ComputeRequest(
            input_tensor=input_tensor_proto,
            tensor_name=test_tensor_name
        )
        mock_context = MagicMock()

        self.servicer.ComputeMatMul(request, mock_context)

        mock_context.abort.assert_called_once()
        # Check that the first argument to abort (status_code) is INVALID_ARGUMENT
        self.assertEqual(mock_context.abort.call_args[0][0], grpc.StatusCode.INVALID_ARGUMENT)

    def test_compute_matmul_tensor_name_not_found(self):
        # Ensure the cache is populated with *something*, but not the tensor we'll request
        self.servicer.tensor_slices_cache["some_other_tensor"] = np.ones((10,10), dtype=np.float32)

        input_data_np = np.ones((3,10), dtype=np.float32)
        input_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(input_data_np.shape),
            serialized_data=input_data_np.tobytes(),
            dtype=str(input_data_np.dtype)
        )
        request = tensor_parallel_pb2.ComputeRequest(
            input_tensor=input_tensor_proto,
            tensor_name="non_existent_tensor" # Request a tensor name not in the cache
        )
        mock_context = MagicMock()

        self.servicer.ComputeMatMul(request, mock_context)

        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.NOT_FOUND, # Shard should abort with NOT_FOUND
            unittest.mock.ANY          # We don't need to assert the exact error message string
        )

if __name__ == "__main__":
    unittest.main()
