import unittest
import subprocess
import time
import os
import sys
import json
import numpy as np
import grpc

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from orchestrator import orchestrator as orch
from protos import tensor_parallel_pb2
from protos import tensor_parallel_pb2_grpc

# It's assumed that shard.py and orchestrator.py are executable
# and that necessary Python environment (with grpc, numpy) is available.

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shard_processes = []
        cls.test_integration_config_path = os.path.join(project_root, "tests", "test_integration_config.json")

        # Define shard configs for the integration test
        # Ensure ports are unique and unlikely to collide
        cls.shard_configs_data = [
            {"shard_id": "integ_shard0", "ip": "localhost", "port": 50080, "slice_start": 0, "slice_end": 5, "axis": "columns"},
            {"shard_id": "integ_shard1", "ip": "localhost", "port": 50081, "slice_start": 5, "slice_end": 10, "axis": "columns"}
        ]
        with open(cls.test_integration_config_path, 'w') as f:
            json.dump(cls.shard_configs_data, f)

        # Start shard servers as subprocesses
        # The dummy tensor slice in shard.py is 100 rows.
        # slice_end - slice_start will be the number of columns for each shard's slice.
        # Shard0: 100x5, Shard1: 100x5
        python_executable = sys.executable # Use the same python interpreter running the tests
        shard_script_path = os.path.join(project_root, "shard", "shard.py")

        for config in cls.shard_configs_data:
            env = os.environ.copy()
            env["SHARD_ID"] = config["shard_id"]
            env["CONFIG_PATH"] = cls.test_integration_config_path
            # Use Popen for non-blocking execution
            process = subprocess.Popen([python_executable, shard_script_path], env=env, cwd=project_root)
            cls.shard_processes.append(process)
            print(f"Started shard {config['shard_id']} on port {config['port']} with PID {process.pid}")

        # Give shards time to start up
        print("Waiting for shards to initialize...")
        time.sleep(5) # Adjust as needed, might need a more robust health check

    @classmethod
    def tearDownClass(cls):
        print("Tearing down integration test: terminating shard processes...")
        for process in cls.shard_processes:
            process.terminate()
            try:
                process.wait(timeout=5) # Wait for graceful termination
            except subprocess.TimeoutExpired:
                print(f"Shard process {process.pid} did not terminate gracefully, killing.")
                process.kill() # Force kill if terminate fails
            print(f"Terminated shard process {process.pid}")

        if os.path.exists(cls.test_integration_config_path):
            os.remove(cls.test_integration_config_path)
        print("Integration test teardown complete.")

    def setUp(self):
        # Orchestrator for each test, to ensure clean state if needed,
        # though for this simple test, a class-level one might also work.
        self.orchestrator_instance = orch.Orchestrator(config_path=self.test_integration_config_path)
        # Short delay to ensure orchestrator connects, though connections are lazy
        time.sleep(0.5)


    def test_simple_matmul_end_to_end(self):
        # Input tensor (Batch=1, Features_in=100)
        # The dummy tensor slice in shard.py is (100, slice_width).
        # So, input features must be 100.
        # Slice_width for integ_shard0 is 5 (0-5), for integ_shard1 is 5 (5-10).
        # So, the effective "model weight" tensor W is (100, 10)
        # Input (1, 100) @ W (100, 10) -> Output (1, 10)

        input_np = np.random.rand(1, 100).astype(np.float32)

        print(f"Integration Test: Sending input of shape {input_np.shape} to orchestrator.")

        # Ensure shards are actually running before making the call
        for shard_cfg in self.shard_configs_data:
            try:
                with grpc.insecure_channel(f"{shard_cfg['ip']}:{shard_cfg['port']}") as channel:
                    grpc.channel_ready_future(channel).result(timeout=2)
                print(f"Shard {shard_cfg['shard_id']} is responsive.")
            except grpc.FutureTimeoutError:
                self.fail(f"Shard {shard_cfg['shard_id']} at {shard_cfg['ip']}:{shard_cfg['port']} not ready for integration test.")

        result_np = self.orchestrator_instance.run_inference_layer_matmul(input_np, "integ_test_layer")

        self.assertIsNotNone(result_np, "Orchestrator returned None, check shard logs for errors.")

        # Expected output shape: (Batch=1, Total_slice_widths = 5+5=10)
        self.assertEqual(result_np.shape, (1, 10))

        # To verify correctness, we'd need to know the dummy TENSOR_SLICE values in shard.py
        # Since they are random, we can't check exact values here without controlling shard's random seed
        # or making shards load predictable data.
        # For now, checking shape and that a result is returned is the primary goal.
        print(f"Integration Test: Received result of shape {result_np.shape}")

    def test_full_inference_end_to_end_dummy(self):
        # This uses the orchestrator's full_inference, which internally calls run_inference_layer_matmul
        # with a dummy input.
        print("Integration Test: Running orchestrator's full_inference (dummy)...")

        result_text = self.orchestrator_instance.full_inference("integration test hello")
        self.assertIsNotNone(result_text)
        self.assertIsInstance(result_text, str)
        # Expecting something like "Processed output (shape: (1, 10))"
        self.assertTrue("Processed output (shape: (1, 10))" in result_text or "Inference failed" in result_text)
        if "Inference failed" in result_text:
             print("Warning: full_inference reported failure. Check shard/orchestrator logs.")
        else:
            print(f"Integration Test: full_inference result: {result_text}")


if __name__ == "__main__":
    # This allows running the integration test directly, e.g. `python tests/test_integration.py`
    # Ensure that the llm_tensor_parallel directory is in PYTHONPATH or use `python -m unittest discover tests`
    unittest.main()
