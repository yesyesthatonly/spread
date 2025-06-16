import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import os
import sys
import json
import grpc # Required for grpc.RpcError

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from orchestrator import orchestrator as orch # Renamed to avoid conflict
from protos import tensor_parallel_pb2
from protos import tensor_parallel_pb2_grpc

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.test_config_path = os.path.join(project_root, "tests", "test_orchestrator_config.json")
        self.shard_configs_data = [
            {"shard_id": "shard0", "ip": "localhost", "port": 50070, "slice_start": 0, "slice_end": 10, "axis": "columns"},
            {"shard_id": "shard1", "ip": "localhost", "port": 50071, "slice_start": 10, "slice_end": 20, "axis": "columns"}
        ]
        with open(self.test_config_path, 'w') as f:
            json.dump(self.shard_configs_data, f)

        # Patch grpc.insecure_channel and the stub
        self.mock_channel = MagicMock()
        self.mock_stub = MagicMock(spec=tensor_parallel_pb2_grpc.TensorParallelServiceStub)

        self.grpc_insecure_channel_patcher = patch('grpc.insecure_channel', return_value=self.mock_channel)
        self.tensor_parallel_service_stub_patcher = patch('protos.tensor_parallel_pb2_grpc.TensorParallelServiceStub', return_value=self.mock_stub)

        self.mock_grpc_insecure_channel = self.grpc_insecure_channel_patcher.start()
        self.mock_tensor_parallel_service_stub = self.tensor_parallel_service_stub_patcher.start()

        # Instantiate Orchestrator - this will call _load_config and _connect_to_shards
        self.orchestrator_instance = orch.Orchestrator(config_path=self.test_config_path)

    def tearDown(self):
        self.grpc_insecure_channel_patcher.stop()
        self.tensor_parallel_service_stub_patcher.stop()
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_load_config(self):
        self.assertEqual(len(self.orchestrator_instance.shard_configs), 2)
        self.assertEqual(self.orchestrator_instance.shard_configs[0]['shard_id'], 'shard0')

    def test_connect_to_shards(self):
        # _connect_to_shards is called in __init__
        self.mock_grpc_insecure_channel.assert_any_call(f"{self.shard_configs_data[0]['ip']}:{self.shard_configs_data[0]['port']}")
        self.mock_grpc_insecure_channel.assert_any_call(f"{self.shard_configs_data[1]['ip']}:{self.shard_configs_data[1]['port']}")
        self.assertEqual(self.mock_grpc_insecure_channel.call_count, 2)
        self.mock_tensor_parallel_service_stub.assert_any_call(self.mock_channel)
        self.assertEqual(self.mock_tensor_parallel_service_stub.call_count, 2)
        self.assertIn('shard0', self.orchestrator_instance.shard_stubs)
        self.assertIn('shard1', self.orchestrator_instance.shard_stubs)

    def test_prepare_input_tensor(self):
        numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = self.orchestrator_instance._prepare_input_tensor(numpy_array)
        self.assertEqual(list(tensor_proto.dims), [2, 2])
        self.assertEqual(list(tensor_proto.data), [1.0, 2.0, 3.0, 4.0])

    def test_process_output_tensor(self):
        tensor_proto = tensor_parallel_pb2.Tensor(dims=[2,2], data=[1.0, 2.0, 3.0, 4.0])
        numpy_array = self.orchestrator_instance._process_output_tensor(tensor_proto)
        expected_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(numpy_array, expected_array)

    def test_run_inference_layer_matmul_success(self):
        # Input: (Batch=1, Features=4)
        input_np = np.array([[1, 2, 3, 4]], dtype=np.float32)

        # Shard0 processes first half of features, Shard1 second half of features (column sharding of weight matrix)
        # Effective weight matrix W is (Features=4, OutputDim=6)
        # Shard0 has W_slice0 (Features=4, OutputDim_slice0=3)
        # Shard1 has W_slice1 (Features=4, OutputDim_slice1=3)
        # Input (1,4) @ W_slice0 (4,3) -> Output_slice0 (1,3)
        # Input (1,4) @ W_slice1 (4,3) -> Output_slice1 (1,3)

        # Mock responses from shards
        output_slice0_np = np.array([[10, 11, 12]], dtype=np.float32) # Result from shard0
        output_slice1_np = np.array([[13, 14, 15]], dtype=np.float32) # Result from shard1

        response0_proto = tensor_parallel_pb2.ComputeResponse(
            output_tensor=tensor_parallel_pb2.Tensor(dims=list(output_slice0_np.shape), data=output_slice0_np.flatten().tolist())
        )
        response1_proto = tensor_parallel_pb2.ComputeResponse(
            output_tensor=tensor_parallel_pb2.Tensor(dims=list(output_slice1_np.shape), data=output_slice1_np.flatten().tolist())
        )

        # Configure the mock stub to return different responses based on the shard it's simulating
        # The orchestrator iterates through self.shard_stubs dictionary. The order might vary.
        # So, we make the mock return values based on some property of the request if possible,
        # or ensure stubs are different if that's easier.
        # Here, self.mock_stub is a single mock for all stubs created. We need to make it smarter
        # or create separate mocks if self.orchestrator_instance.shard_stubs contained different mocks.

        # Since self.orchestrator_instance.shard_stubs maps shard_id to the *same* mock_stub instance,
        # we use side_effect to control its return value per call.
        self.mock_stub.ComputeMatMul.side_effect = [response0_proto, response1_proto]

        result = self.orchestrator_instance.run_inference_layer_matmul(input_np, "test_layer")

        self.assertIsNotNone(result)
        expected_result = np.concatenate((output_slice0_np, output_slice1_np), axis=1)
        np.testing.assert_array_almost_equal(result, expected_result)

        # Check calls to ComputeMatMul
        self.assertEqual(self.mock_stub.ComputeMatMul.call_count, 2)
        # We can inspect calls if needed:
        # calls = self.mock_stub.ComputeMatMul.call_args_list
        # print(calls[0])
        # print(calls[1])


    def test_run_inference_layer_matmul_one_shard_fails(self):
        input_np = np.array([[1, 2, 3, 4]], dtype=np.float32)

        output_slice0_np = np.array([[10, 11, 12]], dtype=np.float32)
        response0_proto = tensor_parallel_pb2.ComputeResponse(
            output_tensor=tensor_parallel_pb2.Tensor(dims=list(output_slice0_np.shape), data=output_slice0_np.flatten().tolist())
        )

        # Simulate RpcError for the second call
        self.mock_stub.ComputeMatMul.side_effect = [
            response0_proto,
            grpc.RpcError("Shard unavailable") # This will be raised by the mock
        ]

        result = self.orchestrator_instance.run_inference_layer_matmul(input_np, "test_layer_fail")
        self.assertIsNone(result) # Orchestrator's current error handling returns None
        self.assertEqual(self.mock_stub.ComputeMatMul.call_count, 2) # Both shards were attempted

    def test_run_inference_layer_matmul_no_shards(self):
        self.orchestrator_instance.shard_stubs = {} # Remove shards
        input_np = np.array([[1,2,3,4]], dtype=np.float32)
        result = self.orchestrator_instance.run_inference_layer_matmul(input_np, "test_layer_no_shards")
        self.assertIsNone(result)

    def test_full_inference_placeholder(self):
        # This test is more of an integration test, but we can check if it calls run_inference_layer_matmul
        # For now, let's just check that it runs without error and returns a string (as per current dummy impl)

        # Mock run_inference_layer_matmul to avoid actual gRPC calls here for this unit test
        with patch.object(self.orchestrator_instance, 'run_inference_layer_matmul', return_value=np.array([[0.1, 0.2]])) as mock_run_matmul:
            result_text = self.orchestrator_instance.full_inference("hello")
            mock_run_matmul.assert_called_once() # It's called with a dummy input
            self.assertIsInstance(result_text, str)
            self.assertTrue("Processed output" in result_text or "Inference failed" in result_text)


if __name__ == "__main__":
    unittest.main()
