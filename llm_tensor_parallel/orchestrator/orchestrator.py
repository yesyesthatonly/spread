import grpc # For gRPC communication (client-side)
import numpy as np
import json # For loading shard configuration
import os

# Import generated gRPC classes
from protos import tensor_parallel_pb2
from protos import tensor_parallel_pb2_grpc

# Default path to the shard configuration file.
# Assumes orchestrator.py is in orchestrator/ and config is in config/ at the same project level.
# Can be overridden by the CONFIG_PATH environment variable.
CONFIG_PATH = os.environ.get("CONFIG_PATH", "../config/shard_config.json")

class Orchestrator:
    """
    Manages distributed tensor computations across multiple shards.

    The Orchestrator is responsible for:
    - Loading shard configurations.
    - Establishing gRPC connections (stubs) to each shard.
    - Preparing input data and sending it to shards for computation.
    - Receiving processed tensor slices from shards.
    - Aggregating results from shards to form the final output for a layer.
    - (Future) Handling the overall LLM inference pipeline, including tokenization and model layer management.
    """
    def __init__(self, config_path=CONFIG_PATH):
        """
        Initializes the Orchestrator.

        Args:
            config_path (str): Path to the shard configuration JSON file.
        """
        self.shard_stubs = {}  # Dictionary to store gRPC stubs, keyed by shard_id
        self.shard_configs = [] # List to store configurations of all shards
        self._load_config(config_path)
        self._connect_to_shards()

        # Placeholder for future components like tokenizer and model architecture details.
        # self.tokenizer = None  # E.g., Hugging Face tokenizer
        # self.model_layers_config = [] # Configuration describing how model tensors are structured/sharded.
        # Example: self.load_model_tokenizer("gpt2-xl") # To be implemented

    def _load_config(self, config_path):
        """
        Loads shard configurations from a JSON file.

        Args:
            config_path (str): Path to the configuration file.

        Populates `self.shard_configs`.
        Raises:
            FileNotFoundError: If the config file is not found.
            json.JSONDecodeError: If the config file is not valid JSON.
        """
        print(f"Orchestrator: Loading shard configuration from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                self.shard_configs = json.load(f) # Expect a list of shard config dicts
            print(f"Orchestrator: Loaded {len(self.shard_configs)} shard configurations.")
        except Exception as e:
            print(f"Orchestrator: Error loading config from {config_path}: {e}")
            raise # Propagate error to prevent orchestrator from running with invalid config

    def _connect_to_shards(self):
        """
        Establishes gRPC connections (stubs) to all configured shards.

        Populates `self.shard_stubs` with `TensorParallelServiceStub` instances.
        """
        print("Orchestrator: Connecting to shards...")
        for shard_info in self.shard_configs:
            shard_id = shard_info["shard_id"]
            address = f"{shard_info['ip']}:{shard_info['port']}" # Construct address from config
            try:
                # Create an insecure gRPC channel to the shard's address.
                # In a production environment, secure channels (TLS) should be used.
                channel = grpc.insecure_channel(address)

                # TODO: Add a more robust health check here.
                # For example, try to connect with a timeout or call a dummy health check RPC on the shard.
                # grpc.channel_ready_future(channel).result(timeout=5) # Example of a blocking wait

                # Create a gRPC stub for the TensorParallelService.
                # This stub is used to make RPC calls to the shard.
                self.shard_stubs[shard_id] = tensor_parallel_pb2_grpc.TensorParallelServiceStub(channel)
                print(f"Orchestrator: Created gRPC stub for shard {shard_id} at {address}")
            except Exception as e:
                # This basic error handling logs and continues, meaning some shards might be unavailable.
                # More sophisticated error handling (e.g., retries, marking shard as unhealthy) could be added.
                print(f"Orchestrator: Error connecting to shard {shard_id} at {address}: {e}")
        print(f"Orchestrator: Attempted to connect to {len(self.shard_stubs)} out of {len(self.shard_configs)} configured shards.")

    def _prepare_input_tensor(self, numpy_array):
        """
        Serializes a NumPy array into a protobuf Tensor message.

        Args:
            numpy_array (np.ndarray): The input tensor.

        Returns:
            tensor_parallel_pb2.Tensor: The protobuf Tensor message.
        """
        return tensor_parallel_pb2.Tensor(
            dims=list(numpy_array.shape),       # Set tensor dimensions
            data=numpy_array.flatten().tolist() # Flatten data and convert to list for protobuf
        )

    def _process_output_tensor(self, tensor_proto):
        """
        Deserializes a protobuf Tensor message into a NumPy array.

        Args:
            tensor_proto (tensor_parallel_pb2.Tensor): The protobuf Tensor message.

        Returns:
            np.ndarray: The deserialized tensor as a NumPy array.
        """
        dims = list(tensor_proto.dims)
        data = np.array(tensor_proto.data, dtype=np.float32).reshape(dims)
        return data

    def run_inference_layer_matmul(self, input_numpy_array, layer_id="layer_0"):
        """
        Performs a distributed matrix multiplication for one layer.

        This method sends the `input_numpy_array` to all configured shards. Each shard
        is expected to perform `input_numpy_array @ local_weight_slice`.
        The results (output slices) are then gathered and concatenated.

        This implementation assumes:
        - The weight matrix (W) for the layer is column-sharded across the shards.
          (i.e., each shard holds a slice W_i, where W = [W_0, W_1, ..., W_n]).
        - The input `input_numpy_array` (A) is sent fully to each shard.
        - The result A @ W is obtained by concatenating [A@W_0, A@W_1, ..., A@W_n] along axis 1.

        Args:
            input_numpy_array (np.ndarray): The input tensor for the layer.
            layer_id (str, optional): Identifier for the layer (for logging/future use).

        Returns:
            np.ndarray or None: The concatenated result tensor if successful, otherwise None.
        """
        print(f"Orchestrator: Running distributed MatMul for {layer_id} with input shape {input_numpy_array.shape}")
        if not self.shard_stubs:
            print("Orchestrator: No shards connected or available. Cannot perform MatMul.")
            return None

        # Serialize the input NumPy array to a protobuf Tensor message for gRPC.
        input_tensor_proto = self._prepare_input_tensor(input_numpy_array)

        results_from_shards = [] # List to store {"shard_id": id, "output_slice": np.array} dicts

        # Iterate over available shard stubs and send the ComputeMatMul request.
        for shard_id, stub in self.shard_stubs.items():
            print(f"Orchestrator: Sending input to shard {shard_id} for {layer_id}...")
            try:
                # Create the gRPC request message.
                request = tensor_parallel_pb2.ComputeRequest(
                    input_tensor=input_tensor_proto,
                    # model_name="gpt2-xl", # Optional: Can be used by shards to select model-specific logic
                    # tensor_name=f"{layer_id}_weights" # Optional: Can identify the specific tensor/layer
                )
                # Make the RPC call to the shard's ComputeMatMul method with a timeout.
                response = stub.ComputeMatMul(request, timeout=10) # 10-second timeout

                # Deserialize the received output slice (protobuf Tensor) back to a NumPy array.
                output_slice = self._process_output_tensor(response.output_tensor)
                results_from_shards.append({"shard_id": shard_id, "output_slice": output_slice})
                print(f"Orchestrator: Received output slice from {shard_id} of shape {output_slice.shape}")

            except grpc.RpcError as e:
                # Handle gRPC errors (e.g., shard unavailable, computation error on shard).
                print(f"Orchestrator: Error calling ComputeMatMul on shard {shard_id}: {e.code()} - {e.details()}")
                # Current simple error handling: if any shard fails, the whole operation fails.
                # More advanced strategies could include retries or partial results.
                return None

        if not results_from_shards:
            print("Orchestrator: No results received from any shards.")
            return None

        # Order the received slices correctly before concatenation.
        # The order is determined by `slice_start` in the shard configuration.
        # This ensures that slices like [A@W_0, A@W_1, A@W_2] are combined in the correct order.

        # Create a map of shard_id to its output slice for easy lookup.
        shard_output_map = {res["shard_id"]: res["output_slice"] for res in results_from_shards}

        # Sort the original shard configurations by their `slice_start` value.
        # This defines the canonical order for assembling the final tensor.
        sorted_shards_info = sorted(self.shard_configs, key=lambda s_conf: s_conf["slice_start"])

        ordered_slices = []
        for s_info in sorted_shards_info:
            s_id = s_info["shard_id"]
            if s_id in shard_output_map:
                ordered_slices.append(shard_output_map[s_id])
            else:
                # This indicates a shard listed in the config did not return a result (or failed).
                # If strict completion is required, this should be an error.
                # For now, it prints a warning. If the earlier loop returned None on error, this path might not be hit.
                print(f"Warning: Orchestrator - No output found for shard {s_id} which was in the configuration. It might have failed.")
                # Depending on policy, might need to return None here if a shard is missing.
                # For now, we proceed with available slices. If this leads to an empty list, it's handled below.

        if not ordered_slices:
            print("Orchestrator: No ordered slices available for concatenation (all shards might have failed or returned no data).")
            return None

        # Concatenate the ordered slices.
        # For column-wise sharding of weights (W = [W0 W1 W2]), outputs (A*W0, A*W1, A*W2)
        # should be concatenated along axis 1 (columns) to reconstruct the full A*W.
        try:
            final_result = np.concatenate(ordered_slices, axis=1)
            print(f"Orchestrator: Concatenated result shape: {final_result.shape}")
            return final_result
        except ValueError as e:
            # This can happen if slices have incompatible shapes for concatenation along axis 1.
            print(f"Orchestrator: Error concatenating result slices: {e}")
            for i, s_arr in enumerate(ordered_slices):
                print(f"  Slice {i} (from shard {sorted_shards_info[i]['shard_id'] if i < len(sorted_shards_info) else 'N/A'}) shape: {s_arr.shape}")
            return None


    def full_inference(self, text_input):
        """
        Placeholder for a full LLM inference pipeline.

        In a complete system, this would involve:
        1. Tokenization of `text_input`.
        2. Passing token embeddings through multiple model layers.
           - For sharded layers (like large MatMuls), it would call `run_inference_layer_matmul`.
           - Other layers (activations, normalization) might be run locally or also distributed.
        3. Detokenization of the final output tokens to produce text.

        Args:
            text_input (str): The input text for the LLM.

        Returns:
            str: The generated text result (currently a dummy string indicating output shape).
        """
        print(f"Orchestrator: Received text input for full_inference: '{text_input}'")

        # DUMMY IMPLEMENTATION:
        # Simulates processing for one layer using a randomly generated input activation.
        # This input activation's shape (1, 100) is chosen to be compatible with the
        # dummy TENSOR_SLICE in shard.py, which is (100, slice_width).
        # (batch_size=1, features_in=100)
        dummy_input_activation = np.random.rand(1, 100).astype(np.float32)

        print(f"Orchestrator: Using DUMMY input activation of shape {dummy_input_activation.shape} for one layer.")

        # Call the distributed matrix multiplication for this dummy layer.
        layer_output = self.run_inference_layer_matmul(dummy_input_activation, layer_id="dummy_llm_layer_1")

        if layer_output is not None:
            # In a real scenario, `layer_output` would feed into the next LLM layer.
            print(f"Orchestrator: Dummy layer output computed. Shape: {layer_output.shape}")
            # Here, we just return a string indicating success and the shape of the result.
            return f"Processed output (shape: {layer_output.shape})"
        else:
            # If `run_inference_layer_matmul` failed (e.g., a shard error).
            return "Inference failed due to error in distributed MatMul."

def main():
    """
    Main function to demonstrate Orchestrator functionality.
    This function initializes and runs the Orchestrator with a sample inference call.
    It expects shard servers to be running and configured in `shard_config.json`.
    """
    # Construct the absolute path to the config file, assuming it's in ../config/ relative to this script.
    # This makes the script runnable from different working directories if this file is invoked directly.
    config_file_path = os.path.join(os.path.dirname(__file__), "..", "config", "shard_config.json")

    print(f"Orchestrator main: Using config file at {config_file_path}")
    orchestrator_instance = Orchestrator(config_path=config_file_path)

    # A short delay to give gRPC channels time to connect, especially if shards were just started.
    # In a robust system, readiness probes or explicit connection checks would be better.
    print("Orchestrator main: Waiting a few seconds for shards to be ready (simulated)...")
    import time # Local import for this main function
    time.sleep(5) # seconds

    # Perform a dummy full inference.
    inference_result = orchestrator_instance.full_inference("Hello LLM world!")
    print(f"Orchestrator main: Full inference result: {inference_result}")

    # Example of directly testing the `run_inference_layer_matmul` method:
    # This requires an input compatible with the dummy shard tensor slices (100 features_in).
    # test_input_data = np.random.rand(1, 100).astype(np.float32) # Batch of 1, 100 features_in
    # print(f"Orchestrator main: Testing run_inference_layer_matmul with input shape {test_input_data.shape}...")
    # layer_output_result = orchestrator_instance.run_inference_layer_matmul(test_input_data, "direct_test_layer")
    # if layer_output_result is not None:
    #     print(f"Orchestrator main: Direct test layer output shape: {layer_output_result.shape}")
    # else:
    #     print("Orchestrator main: Direct test layer call failed.")


if __name__ == "__main__":
    # This block executes when the script is run directly (e.g., `python orchestrator.py`)
    main()
