import unittest
import subprocess
import time
import os
import sys
import json
import numpy as np
import grpc
import torch # Added for PyTorch operations

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from orchestrator import orchestrator as orch
from protos import tensor_parallel_pb2 # Not directly used, but good for context
from protos import tensor_parallel_pb2_grpc # For grpc.channel_ready_future

# It's assumed that shard.py and orchestrator.py are executable
# and that necessary Python environment (with grpc, numpy, torch, transformers) is available.

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shard_processes = []
        cls.test_model_name = "gpt2" # Use standard small GPT-2 for integration tests
        cls.test_integration_config_path = os.path.join(project_root, "tests", "test_integration_config.json")

        # GPT-2 'n_embd' (hidden size) is 768.
        # MLP 'c_fc' layer has output features 4 * n_embd = 3072.
        # We will shard this dimension (3072) across 2 shards.
        # This config targets the 'mlp.c_fc.weight' tensor primarily for sharding.
        # Shards will attempt to apply these slices to other shardeable tensors too.
        cls.gpt2_hidden_size = 768
        cls.gpt2_mlp_intermediate_size = 4 * cls.gpt2_hidden_size # 3072

        cls.shard_configs_data = [
            {
                "shard_id": "integ_shard0", "ip": "localhost", "port": 50080,
                "slice_start": 0, "slice_end": cls.gpt2_mlp_intermediate_size // 2, # 0 to 1536
                "axis": "columns", "model_name": cls.test_model_name
            },
            {
                "shard_id": "integ_shard1", "ip": "localhost", "port": 50081,
                "slice_start": cls.gpt2_mlp_intermediate_size // 2, # 1536
                "slice_end": cls.gpt2_mlp_intermediate_size,       # 1536 to 3072
                "axis": "columns", "model_name": cls.test_model_name
            }
        ]
        with open(cls.test_integration_config_path, 'w') as f:
            json.dump(cls.shard_configs_data, f)

        python_executable = sys.executable
        shard_script_path = os.path.join(project_root, "shard", "shard.py")

        for config in cls.shard_configs_data:
            env = os.environ.copy()
            env["SHARD_ID"] = config["shard_id"]
            env["CONFIG_PATH"] = cls.test_integration_config_path
            # To ensure transformers download to a predictable place if not cached by host user
            env["HF_HOME"] = os.path.join(project_root, ".test_cache", "huggingface")
            env["TRANSFORMERS_CACHE"] = os.path.join(project_root, ".test_cache", "huggingface", "models")

            process = subprocess.Popen([python_executable, shard_script_path], env=env, cwd=project_root)
            cls.shard_processes.append(process)
            print(f"Started shard {config['shard_id']} on port {config['port']} with PID {process.pid}")

        print("Waiting for shards to initialize (load model, slices)...")
        # Increased wait time as shards now load models from Hugging Face hub if not cached
        # A more robust readiness check would involve trying to connect or a health check RPC.
        cls.wait_for_shards_ready(timeout=60)


    @classmethod
    def wait_for_shards_ready(cls, timeout=60):
        start_time = time.time()
        for shard_cfg in cls.shard_configs_data:
            print(f"Checking readiness of shard {shard_cfg['shard_id']} at {shard_cfg['ip']}:{shard_cfg['port']}...")
            while True:
                try:
                    with grpc.insecure_channel(f"{shard_cfg['ip']}:{shard_cfg['port']}") as channel:
                        grpc.channel_ready_future(channel).result(timeout=2) # Short timeout for individual check
                    print(f"Shard {shard_cfg['shard_id']} is responsive.")
                    break # Shard is ready
                except grpc.FutureTimeoutError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Timeout waiting for shard {shard_cfg['shard_id']} to become ready.")
                    time.sleep(1) # Wait a bit before retrying


    @classmethod
    def tearDownClass(cls):
        print("Tearing down integration test: terminating shard processes...")
        for process in cls.shard_processes:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Shard process {process.pid} did not terminate gracefully, killing.")
                process.kill()
            print(f"Terminated shard process {process.pid}")

        if os.path.exists(cls.test_integration_config_path):
            os.remove(cls.test_integration_config_path)
        # Clean up test cache if necessary, though typically left for subsequent runs
        # import shutil
        # test_cache_dir = os.path.join(project_root, ".test_cache")
        # if os.path.exists(test_cache_dir):
        #     shutil.rmtree(test_cache_dir)
        print("Integration test teardown complete.")

    def setUp(self):
        # Instantiate Orchestrator. It will load the "gpt2" model.
        self.orchestrator_instance = orch.Orchestrator(
            config_path=self.test_integration_config_path,
            model_name=self.test_model_name
        )
        # Allow some time for the orchestrator to initialize its model
        # (though model loading is now synchronous in __init__)
        time.sleep(1)

    def test_mlp_layer_numerical_correctness(self):
        print("Integration Test: Verifying MLP layer numerical correctness...")
        # Use the orchestrator's loaded model for reference computation
        full_model = self.orchestrator_instance.model
        device = self.orchestrator_instance.device

        # Target the first MLP block (layer 0)
        layer_idx = 0
        block0 = full_model.transformer.h[layer_idx]

        # Create a sample input activation tensor
        # Shape: (batch_size, sequence_length, hidden_size)
        # Using 1 token for simplicity in verifying MatMul.
        batch_size = 1
        seq_len = 1
        sample_input_activation_torch = torch.randn(batch_size, seq_len, self.gpt2_hidden_size, device=device).float()

        # --- Reference MLP computation (non-sharded, on orchestrator's full model) ---
        # 1. LayerNorm before MLP (ln_2)
        ln2_output_ref = block0.ln_2(sample_input_activation_torch)

        # 2. First MLP linear layer (c_fc)
        fc_output_ref = block0.mlp.c_fc(ln2_output_ref)

        # 3. Activation function (GELU)
        activated_output_ref = torch.nn.functional.gelu(fc_output_ref, approximate="tanh")

        # 4. Second MLP linear layer (c_proj)
        proj_output_ref = block0.mlp.c_proj(activated_output_ref)
        # proj_output_ref is the output of one MLP block, shape (batch, seq, hidden_size)

        # --- Sharded MLP computation (via orchestrator) ---
        # Reshape input for run_inference_layer_matmul: (batch*seq, hidden_size)
        ln2_output_ref_np_reshaped = ln2_output_ref.view(-1, self.gpt2_hidden_size).cpu().detach().numpy()

        # 1. Sharded c_fc layer
        fc_tensor_name = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
        fc_sharded_output_np = self.orchestrator_instance.run_inference_layer_matmul(
            ln2_output_ref_np_reshaped,
            layer_id=f"integ_test_layer_{layer_idx}_mlp_fc",
            tensor_name_for_shards=fc_tensor_name
        )
        self.assertIsNotNone(fc_sharded_output_np, f"Sharded MatMul failed for {fc_tensor_name}")

        # Add bias (orchestrator side)
        fc_bias_np = block0.mlp.c_fc.bias.data.cpu().detach().numpy()
        fc_sharded_with_bias_np = fc_sharded_output_np + fc_bias_np

        # 2. Activation (orchestrator side)
        activated_sharded_torch = torch.nn.functional.gelu(torch.from_numpy(fc_sharded_with_bias_np), approximate="tanh")
        activated_sharded_np = activated_sharded_torch.numpy() # Shape: (batch*seq, 4*hidden_size)

        # 3. Sharded c_proj layer
        proj_tensor_name = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        proj_sharded_output_np = self.orchestrator_instance.run_inference_layer_matmul(
            activated_sharded_np, # Input is output of GELU
            layer_id=f"integ_test_layer_{layer_idx}_mlp_proj",
            tensor_name_for_shards=proj_tensor_name
        )
        self.assertIsNotNone(proj_sharded_output_np, f"Sharded MatMul failed for {proj_tensor_name}")

        # Add bias (orchestrator side)
        proj_bias_np = block0.mlp.c_proj.bias.data.cpu().detach().numpy()
        proj_sharded_with_bias_np = proj_sharded_output_np + proj_bias_np

        # Reshape sharded output to match reference: (batch, seq, hidden_size)
        proj_sharded_output_reshaped_np = proj_sharded_with_bias_np.reshape(batch_size, seq_len, self.gpt2_hidden_size)

        # --- Comparison ---
        print("Reference MLP output shape:", proj_output_ref.shape)
        print("Sharded MLP output shape:", proj_sharded_output_reshaped_np.shape)

        np.testing.assert_allclose(
            proj_output_ref.cpu().detach().numpy(),
            proj_sharded_output_reshaped_np,
            rtol=1e-5, atol=1e-5 # Adjust tolerance as needed
        )
        print("Integration Test: MLP layer numerical correctness verified.")


    def test_minimal_text_generation_e2e(self):
        print("Integration Test: Running minimal end-to-end text generation...")
        prompt = "Hello"
        # max_length in full_inference is number of *new* tokens
        # Total length will be len(prompt_tokens) + max_length
        max_new_tokens = 3

        # Ensure shards are responsive before this test
        # (already done in setUpClass, but an extra check here if running test standalone)
        try:
            TestIntegration.wait_for_shards_ready(timeout=10) # Shorter timeout for re-check
        except TimeoutError as e:
            self.fail(f"Shards not ready for text generation test: {e}")

        output_text = self.orchestrator_instance.full_inference(prompt, max_length=max_new_tokens)

        self.assertIsInstance(output_text, str)
        self.assertTrue(len(output_text) > 0, "Generated text is empty.")
        # A simple check: output should be longer than input if tokens were added.
        # This isn't always true if tokenizer behavior is complex or EOS is immediate.
        # A better check might be that it's not *just* the prompt.
        if prompt not in output_text : # If prompt is not part of output (e.g. special tokens stripped)
             print(f"Warning: Prompt '{prompt}' not found in output '{output_text}'. This might be okay depending on tokenizer.")
        else:
            self.assertTrue(len(output_text) > len(prompt),
                            f"Generated text ('{output_text}') is not longer than prompt ('{prompt}'). No new tokens?")

        print(f"Integration Test: Minimal text generation completed. Prompt: '{prompt}', Output: '{output_text}'")


if __name__ == "__main__":
    unittest.main()
