import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys
import json
import grpc # Required for grpc.StatusCode for one of the tests

# Add project root to sys.path to allow importing protos and shard
# This assumes tests are run from the root of the llm_tensor_parallel directory or that the path is otherwise managed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from protos import tensor_parallel_pb2
# Ensure shard module can be found. If shard.py is in shard/, and protos is in protos/
# and this test is in tests/, this path adjustment is crucial.
from shard import shard

class TestShard(unittest.TestCase):

    def setUp(self):
        # Create a dummy config file for testing
        self.test_config_path = os.path.join(project_root, "tests", "test_shard_config.json")
        self.shard0_config = {
            "shard_id": "test_shard0", "ip": "localhost", "port": 50090,
            "slice_start": 0, "slice_end": 10, "axis": "columns"
        }
        self.shard1_config_rows = {
            "shard_id": "test_shard1_rows", "ip": "localhost", "port": 50091,
            "slice_start": 0, "slice_end": 5, "axis": "rows"
        }
        with open(self.test_config_path, 'w') as f:
            json.dump([self.shard0_config, self.shard1_config_rows], f)

        # Patch environment variables for shard loading
        self.env_patcher_shard_id = patch.dict(os.environ, {"SHARD_ID": "test_shard0", "CONFIG_PATH": self.test_config_path})
        self.env_patcher_shard_id.start()

        # Instantiate the servicer
        self.servicer = shard.TensorParallelServiceServicer()
        # Reset global TENSOR_SLICE before each test that relies on load_tensor_slice
        shard.TENSOR_SLICE = None


    def tearDown(self):
        self.env_patcher_shard_id.stop()
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        shard.TENSOR_SLICE = None # Ensure cleanup

    def test_load_tensor_slice_columns(self):
        # Test loading a column slice
        shard.load_tensor_slice(shard_id="test_shard0", config_path=self.test_config_path)
        self.assertIsNotNone(shard.TENSOR_SLICE)
        # Dummy tensor in shard.py is 100 rows. Slice is cols 0-10.
        self.assertEqual(shard.TENSOR_SLICE.shape, (100, 10))

    def test_load_tensor_slice_rows(self):
        # Test loading a row slice
        # Need to change SHARD_ID env var for this test
        with patch.dict(os.environ, {"SHARD_ID": "test_shard1_rows", "CONFIG_PATH": self.test_config_path}):
            shard.load_tensor_slice(shard_id="test_shard1_rows", config_path=self.test_config_path)
        self.assertIsNotNone(shard.TENSOR_SLICE)
        # Dummy tensor in shard.py for rows is slice_end-slice_start (5) rows, 100 cols.
        self.assertEqual(shard.TENSOR_SLICE.shape, (5, 100))

    def test_compute_matmul_column_slice(self):
        # Setup a known tensor slice for consistent testing
        # Slice is (K, N_slice) = (20, 10)
        shard.TENSOR_SLICE = np.ones((20, 10), dtype=np.float32)

        # Input tensor (B, K) = (3, 20)
        input_data = np.full((3, 20), 2.0, dtype=np.float32)

        input_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(input_data.shape),
            data=input_data.flatten().tolist()
        )
        request = tensor_parallel_pb2.ComputeRequest(input_tensor=input_tensor_proto)

        mock_context = MagicMock()
        response = self.servicer.ComputeMatMul(request, mock_context)

        self.assertIsNotNone(response.output_tensor)
        output_dims = list(response.output_tensor.dims)
        output_data = np.array(response.output_tensor.data).reshape(output_dims)

        # Expected_output = Input (3x20) @ Slice (20x10) = Result (3x10)
        # All ones in slice, all 2.0 in input. Sum of 20 elements of 2.0 = 40.0
        expected_output = np.full((3, 10), 40.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(output_data, expected_output)
        mock_context.abort.assert_not_called()

    def test_compute_matmul_value_error(self):
        # Test matmul with incompatible shapes
        shard.TENSOR_SLICE = np.ones((5, 5), dtype=np.float32) # K=5
        input_data = np.ones((3, 4), dtype=np.float32)       # K=4, different

        input_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(input_data.shape),
            data=input_data.flatten().tolist()
        )
        request = tensor_parallel_pb2.ComputeRequest(input_tensor=input_tensor_proto)
        mock_context = MagicMock()

        self.servicer.ComputeMatMul(request, mock_context)
        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, # This was the missing import
            unittest.mock.ANY # The error message string
        )

    def test_compute_matmul_tensor_not_loaded(self):
        shard.TENSOR_SLICE = None # Ensure tensor is not loaded
        input_data = np.ones((3,4), dtype=np.float32)
        input_tensor_proto = tensor_parallel_pb2.Tensor(dims=list(input_data.shape), data=input_data.flatten().tolist())
        request = tensor_parallel_pb2.ComputeRequest(input_tensor=input_tensor_proto)
        mock_context = MagicMock()

        self.servicer.ComputeMatMul(request, mock_context)
        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION, "Tensor slice not loaded."
        )

if __name__ == "__main__":
    # Need to make sure imports work correctly when run directly
    # This is often tricky with relative imports and project structure.
    # Running 'python -m unittest discover tests' from project root is preferred.
    unittest.main()
