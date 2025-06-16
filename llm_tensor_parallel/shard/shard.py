import grpc
from concurrent import futures # For creating a thread pool for the server
import time
import numpy as np
import json
import os

# Import generated gRPC classes
from protos import tensor_parallel_pb2
from protos import tensor_parallel_pb2_grpc

# Global variable to store this shard's tensor slice.
# In a more complex application, this might be part of a class or a more sophisticated cache.
TENSOR_SLICE = None
# SHARD_ID is obtained from environment variable, defaulting to "shard0".
# This ID is used to find this shard's specific configuration in the config file.
SHARD_ID = os.environ.get("SHARD_ID", "shard0")
# CONFIG_PATH is obtained from environment variable, defaulting to a relative path.
# This JSON file contains connection info and slicing details for all shards.
CONFIG_PATH = os.environ.get("CONFIG_PATH", "../config/shard_config.json")

def load_tensor_slice(shard_id, config_path, model_name="gpt2-xl", tensor_name_key="some_tensor_key"):
    """
    Loads a tensor slice for this shard based on its ID and configuration.

    Args:
        shard_id (str): The unique identifier for this shard.
        config_path (str): Path to the JSON configuration file.
        model_name (str, optional): Name of the model (for future use with real model loading).
        tensor_name_key (str, optional): Key to identify a specific tensor (for future use).

    This is currently a DUMMY IMPLEMENTATION. It generates a random tensor slice
    based on parameters found in the config file for this shard_id.
    In a real system, this function would load a part of a real model's tensor.
    """
    global TENSOR_SLICE # Modifies the global TENSOR_SLICE variable

    # This is a placeholder for actual model loading and slicing.
    # Steps for a real implementation:
    # 1. Read `config_path` to find this shard's specific info (e.g., slice_start, slice_end, axis).
    # 2. Load the specified model (e.g., `model_name` from Hugging Face Transformers).
    # 3. Extract the relevant tensor (e.g., a weight matrix identified by `tensor_name_key`).
    # 4. Slice this tensor according to this shard's configuration.
    # 5. Store the resulting slice in the global `TENSOR_SLICE`.

    print(f"Shard {shard_id}: Loading tensor slice (dummy implementation)...")
    try:
        # Attempt to read the main configuration file
        with open(config_path, 'r') as f:
            config = json.load(f) # `config` is a list of shard configurations

        # Find the specific configuration for this shard_id
        shard_config = next((s_conf for s_conf in config if s_conf["shard_id"] == shard_id), None)

        if shard_config:
            slice_start = shard_config["slice_start"]
            slice_end = shard_config["slice_end"]

            # For this dummy implementation, we define a fixed dimension for the "model's full tensor"
            # If sharding by columns, this is the number of rows.
            # If sharding by rows, this is the number of columns.
            # This value (100) must be consistent with the orchestrator's dummy input.
            common_dimension_size = 100

            if shard_config["axis"] == "columns":
                # When sharding by columns, the slice is tensor[:, slice_start:slice_end]
                # So, the number of rows is `common_dimension_size`, and columns is `slice_end - slice_start`.
                TENSOR_SLICE = np.random.rand(common_dimension_size, slice_end - slice_start).astype(np.float32)
            elif shard_config["axis"] == "rows":
                # When sharding by rows, the slice is tensor[slice_start:slice_end, :]
                # So, the number of rows is `slice_end - slice_start`, and columns is `common_dimension_size`.
                TENSOR_SLICE = np.random.rand(slice_end - slice_start, common_dimension_size).astype(np.float32)
            else:
                # Invalid axis configuration
                raise ValueError(f"Unknown axis for sharding: {shard_config['axis']}")

            print(f"Shard {shard_id}: Loaded DUMMY tensor slice of shape {TENSOR_SLICE.shape} for axis '{shard_config['axis']}' (indices {slice_start} to {slice_end}).")
        else:
            # Configuration for this specific shard_id was not found.
            print(f"Shard {shard_id}: Configuration not found in {config_path}.")
            # Fallback to a default small random tensor if no specific config is found.
            TENSOR_SLICE = np.random.rand(10,10).astype(np.float32)

    except Exception as e:
        # Catch any errors during loading (e.g., file not found, JSON parsing error)
        print(f"Error loading tensor slice for shard {shard_id}: {e}")
        # Fallback to a default small random tensor in case of error.
        TENSOR_SLICE = np.random.rand(10,10).astype(np.float32)


class TensorParallelServiceServicer(tensor_parallel_pb2_grpc.TensorParallelServiceServicer):
    """
    gRPC service implementation for tensor parallelism.
    This class handles incoming RPC calls defined in tensor_parallel.proto.
    """
    def ComputeMatMul(self, request, context):
        """
        Computes a matrix multiplication slice.
        Input tensor is multiplied by this shard's TENSOR_SLICE.
        """
        print(f"Shard {SHARD_ID}: Received ComputeMatMul request.")

        # Check if the tensor slice has been loaded
        if TENSOR_SLICE is None:
            print(f"Shard {SHARD_ID}: Error - Tensor slice not loaded.")
            # Report an error to the gRPC client (orchestrator)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Tensor slice not loaded.")
            return tensor_parallel_pb2.ComputeResponse() # Return empty response

        input_tensor_proto = request.input_tensor

        # Deserialize the input tensor from protobuf message to a NumPy array
        # The protobuf Tensor message stores dimensions and flattened data.
        input_dims = list(input_tensor_proto.dims)
        try:
            input_data = np.array(input_tensor_proto.data, dtype=np.float32).reshape(input_dims)
        except ValueError as e:
            error_msg = f"Shard {SHARD_ID}: Error deserializing input tensor: {e}. Received dims: {input_dims}, data length: {len(input_tensor_proto.data)}"
            print(error_msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            return tensor_parallel_pb2.ComputeResponse()

        print(f"Shard {SHARD_ID}: Input tensor shape: {input_data.shape}, Local slice shape: {TENSOR_SLICE.shape}")

        # Perform the matrix multiplication: result = input_data @ TENSOR_SLICE
        # This assumes TENSOR_SLICE is the right-hand matrix (e.g., a weight slice W_i)
        # and input_data is the activation matrix (A).
        # If W is column-sharded (axis="columns"), then TENSOR_SLICE is W_i.
        # The operation is A @ W_i, resulting in a slice of the output activation.
        try:
            result_data = np.matmul(input_data, TENSOR_SLICE)
        except ValueError as e:
            # This typically occurs if matrix dimensions are incompatible for multiplication.
            error_msg = f"Matrix multiplication error on shard {SHARD_ID}: {e}. Input shape {input_data.shape}, Slice shape {TENSOR_SLICE.shape}"
            print(error_msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            return tensor_parallel_pb2.ComputeResponse()

        print(f"Shard {SHARD_ID}: Computation complete. Result slice shape: {result_data.shape}")

        # Serialize the output tensor slice from NumPy array to protobuf message
        output_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(result_data.shape),
            data=result_data.flatten().tolist() # Flatten data for protobuf's repeated float field
        )
        return tensor_parallel_pb2.ComputeResponse(output_tensor=output_tensor_proto)

def serve():
    """
    Starts the gRPC server for this shard.
    """
    # Load this shard's tensor slice before the server starts accepting requests.
    # SHARD_ID and CONFIG_PATH are read from environment variables at the top of the file.
    load_tensor_slice(SHARD_ID, CONFIG_PATH)

    # Create a gRPC server with a thread pool for handling requests.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add the implemented servicer (TensorParallelServiceServicer) to the server.
    tensor_parallel_pb2_grpc.add_TensorParallelServiceServicer_to_server(
        TensorParallelServiceServicer(), server
    )

    # Determine the port for this shard.
    # It reads the main config file and finds the port associated with its SHARD_ID.
    port = "50051" # Default port if not found in config or if config is malformed.
    try:
        with open(CONFIG_PATH, 'r') as f:
            config_data = json.load(f) # List of shard configurations

        # Find this shard's specific configuration dict
        shard_config = next((s_conf for s_conf in config_data if s_conf["shard_id"] == SHARD_ID), None)

        if shard_config and "port" in shard_config:
            port = str(shard_config["port"])
        else:
            print(f"Warning: Port not found for shard {SHARD_ID} in {CONFIG_PATH}, using default port {port}.")
    except Exception as e:
        print(f"Error reading port from config file {CONFIG_PATH} for shard {SHARD_ID}: {e}. Using default port {port}.")

    # Start the server on the determined port. '[::]' means listen on all available IPv6 and IPv4 interfaces.
    server.add_insecure_port(f"[::]:{port}")
    print(f"Shard {SHARD_ID} server starting on port {port}...")
    server.start()
    print(f"Shard {SHARD_ID} server started successfully on port {port}. Awaiting requests.")

    # Keep the server running indefinitely (or until interrupted).
    try:
        while True:
            time.sleep(86400)  # Sleep for one day (in seconds)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        print(f"Shard {SHARD_ID} server stopping due to keyboard interrupt...")
        server.stop(0) # 0 is grace period in seconds
        print(f"Shard {SHARD_ID} server stopped.")

if __name__ == "__main__":
    # This block executes when the script is run directly (e.g., `python shard.py`)
    serve()
