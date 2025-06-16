import grpc # For gRPC communication (client-side)
import numpy as np
import json # For loading shard configuration
import os

# Import Hugging Face libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
    def __init__(self, config_path=CONFIG_PATH, model_name="gpt2-xl"):
        """
        Initializes the Orchestrator.

        Args:
            config_path (str): Path to the shard configuration JSON file.
            model_name (str): Name of the Hugging Face model to load (e.g., "gpt2", "gpt2-xl").
        """
        self.shard_stubs = {}  # Dictionary to store gRPC stubs, keyed by shard_id
        self.shard_configs = [] # List to store configurations of all shards
        self._load_config(config_path)
        self._connect_to_shards()

        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_tokenizer() # Load model and tokenizer
        if self.model: # If model was loaded successfully
            self.model.to(self.device) # Move model to the chosen device

        # self.model_layers_config = [] # This might be used later for layer-specific logic

    def _load_model_and_tokenizer(self):
        """
        Loads the Hugging Face model and tokenizer specified by `self.model_name`.
        Sets `self.model` and `self.tokenizer` attributes.
        """
        print(f"Orchestrator: Loading model and tokenizer for '{self.model_name}'...")
        try:
            # Device selection:
            # For now, the orchestrator primarily manages and distributes work.
            # If it were to run some layers itself (e.g., embedding layer, final output layer),
            # moving the model (or parts of it) to CUDA would be beneficial if available.
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add padding token if it doesn't exist (common for GPT-2 models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # If we were fine-tuning, we might also need to update model.config:
                # self.model.config.pad_token_id = self.tokenizer.pad_token_id

            # Load the pre-trained model.
            # In a highly memory-optimized scenario for extremely large models,
            # the orchestrator might only load the model's architecture or metadata here.
            # Shards would then be responsible for loading their respective weight slices.
            # However, for models like GPT-2-XL, loading the full model on the orchestrator
            # is often feasible and simplifies handling of non-sharded layers (e.g., embeddings, LayerNorm).
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            # self.model.to(self.device) # Move model to the selected device if orchestrator runs parts of it
            self.model.eval() # Set the model to evaluation mode (disables dropout, etc.)

            print(f"Orchestrator: Model '{self.model_name}' and tokenizer loaded successfully.")

            # Optional: Print model structure or specific parameter details for debugging.
            # print("Model Config:", self.model.config)
            # print(f"Tokenizer pad token: {self.tokenizer.pad_token}, ID: {self.tokenizer.pad_token_id}")
            # print(f"Model pad token ID: {self.model.config.pad_token_id}")

        except Exception as e:
            print(f"Orchestrator: Error loading model or tokenizer for '{self.model_name}': {e}")
            # This is a critical failure, so re-raise to stop orchestrator initialization.
            raise

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
            dims=list(numpy_array.shape),
            serialized_data=numpy_array.tobytes(), # Use tobytes()
            dtype=str(numpy_array.dtype)          # Store dtype as string
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
        dtype = np.dtype(tensor_proto.dtype) # Convert string back to numpy dtype
        data = np.frombuffer(tensor_proto.serialized_data, dtype=dtype).reshape(dims)
        return data

    def run_inference_layer_matmul(self, input_numpy_array, layer_id="layer_0", tensor_name_for_shards=None):
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
        # The following is the existing code for run_inference_layer_matmul, with the tensor_name highlighted.
        print(f"Orchestrator: Running matmul for {layer_id} (tensor: {tensor_name_for_shards}) with input shape {input_numpy_array.shape}")
        if not self.shard_stubs:
            print("Orchestrator: No shards connected.")
            return None

        input_tensor_proto = self._prepare_input_tensor(input_numpy_array)
        results_from_shards = []

        for shard_id, stub in self.shard_stubs.items():
            print(f"Orchestrator: Sending input to shard {shard_id} for {layer_id} (tensor: {tensor_name_for_shards})...")
            try:
                request = tensor_parallel_pb2.ComputeRequest(
                    input_tensor=input_tensor_proto,
                    model_name=self.model_name,
                    tensor_name=tensor_name_for_shards # Ensure this is passed
                )
                # Increased timeout for potentially larger computations
                response = stub.ComputeMatMul(request, timeout=30)
                output_slice = self._process_output_tensor(response.output_tensor)
                # Store slice_start along with the result for robust sorting
                results_from_shards.append({
                    "shard_id": shard_id,
                    "output_slice": output_slice,
                    "slice_start": next(s_conf["slice_start"] for s_conf in self.shard_configs if s_conf["shard_id"] == shard_id)
                })
                print(f"Orchestrator: Received output slice from {shard_id} of shape {output_slice.shape}")
            except grpc.RpcError as e:
                print(f"Orchestrator: Error calling ComputeMatMul on shard {shard_id} for tensor {tensor_name_for_shards}: {e}")
                return None

        if not results_from_shards:
            print(f"Orchestrator: No results received from shards for {tensor_name_for_shards}.")
            return None

        # Sort results by slice_start to ensure correct concatenation order
        results_from_shards.sort(key=lambda res: res["slice_start"])
        ordered_slices = [res["output_slice"] for res in results_from_shards]

        if not ordered_slices:
            print(f"Orchestrator: No ordered slices to concatenate for {tensor_name_for_shards}.")
            return None

        try:
            # Assuming column-wise sharding of weights, concatenate along axis 1 (columns)
            final_result = np.concatenate(ordered_slices, axis=1)
            print(f"Orchestrator: Concatenated result for {tensor_name_for_shards}, shape: {final_result.shape}")
            return final_result
        except ValueError as e:
            print(f"Orchestrator: Error concatenating results for {tensor_name_for_shards}: {e}")
            for i, s_info in enumerate(results_from_shards): # Use results_from_shards for shard_id here
                print(f"  Shard {s_info['shard_id']} (start: {s_info['slice_start']}) slice shape: {s_info['output_slice'].shape}")
            return None

    def full_inference(self, text_input, max_length=50):
        if not self.model or not self.tokenizer:
            print("Orchestrator: Model or tokenizer not loaded.")
            return "Error: Model not loaded."

        print(f"Orchestrator: Running full inference for input: '{text_input}'")

        # 1. Tokenize Input
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)

        generated_ids = input_ids.clone()

        for _ in range(max_length): # Loop to generate tokens up to max_length
            current_input_ids = generated_ids

            # 2. Get initial embeddings (non-sharded on orchestrator)
            current_sequence_length = current_input_ids.shape[-1]
            position_ids = torch.arange(0, current_sequence_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0)

            hidden_states = self.model.transformer.wte(current_input_ids) + self.model.transformer.wpe(position_ids)
            hidden_states = self.model.transformer.drop(hidden_states) # Apply dropout

            # 3. Iterate Through Model Layers (Transformer Blocks)
            for i, block in enumerate(self.model.transformer.h):
                print(f"Orchestrator: Processing Layer {i}")

                # Layer Norm 1 (Non-sharded)
                ln_1_out = block.ln_1(hidden_states)

                # Attention Block - Currently NON-SHARDED on Orchestrator for simplicity
                # In a future step, QKV projection (c_attn) and output projection (c_proj)
                # within the attention block could be sharded.
                # For now, the whole attention block runs on the orchestrator.
                attn_outputs = block.attn(ln_1_out, use_cache=False)
                attn_output = attn_outputs[0]

                # Residual Connection 1
                hidden_states = hidden_states + attn_output

                # Layer Norm 2 (Non-sharded)
                ln_2_out = block.ln_2(hidden_states)

                # MLP Block (c_fc and c_proj ARE SHARDED)
                # MLP First Linear Layer (c_fc)
                mlp_fc_tensor_name = f"transformer.h.{i}.mlp.c_fc.weight"

                batch_size, seq_len, hidden_size = ln_2_out.shape # Get current shapes
                ln_2_out_reshaped_np = ln_2_out.view(-1, hidden_size).cpu().numpy()

                mlp_fc_sharded_output_np = self.run_inference_layer_matmul(
                    ln_2_out_reshaped_np,
                    layer_id=f"layer_{i}_mlp_fc",
                    tensor_name_for_shards=mlp_fc_tensor_name
                )
                if mlp_fc_sharded_output_np is None:
                    return f"Error: Sharded MatMul failed for MLP FC at layer {i}."

                mlp_fc_bias = self.model.transformer.h[i].mlp.c_fc.bias.data.cpu().numpy()
                mlp_fc_with_bias_np = mlp_fc_sharded_output_np + mlp_fc_bias

                # GELU Activation (Non-sharded)
                mlp_activated_torch = torch.nn.functional.gelu(
                    torch.from_numpy(mlp_fc_with_bias_np), approximate="tanh"
                ).to(self.device)

                # MLP Second Linear Layer (c_proj)
                mlp_proj_tensor_name = f"transformer.h.{i}.mlp.c_proj.weight"
                mlp_activated_np = mlp_activated_torch.cpu().numpy() # Already effectively reshaped

                mlp_proj_sharded_output_np = self.run_inference_layer_matmul(
                    mlp_activated_np,
                    layer_id=f"layer_{i}_mlp_proj",
                    tensor_name_for_shards=mlp_proj_tensor_name
                )
                if mlp_proj_sharded_output_np is None:
                    return f"Error: Sharded MatMul failed for MLP Proj at layer {i}."

                mlp_proj_bias = self.model.transformer.h[i].mlp.c_proj.bias.data.cpu().numpy()
                mlp_proj_with_bias_np = mlp_proj_sharded_output_np + mlp_proj_bias

                mlp_output_torch = torch.from_numpy(mlp_proj_with_bias_np).view(batch_size, seq_len, hidden_size).to(self.device)

                # Residual Connection 2
                hidden_states = hidden_states + mlp_output_torch

            # After all transformer blocks, apply final Layer Norm (Non-sharded)
            hidden_states = self.model.transformer.ln_f(hidden_states)

            # 4. Get Logits (Non-sharded) for the last token
            logits = self.model.lm_head(hidden_states[:, -1, :])

            # 5. Generate Next Token ID (Greedy decoding, Non-sharded)
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)

            # 6. Append to generated_ids and check for EOS
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                print("Orchestrator: EOS token generated.")
                break

        # 7. Detokenize Output
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Orchestrator: Generated text: '{output_text}'")
        return output_text


    def full_inference_old_placeholder(self, text_input):
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
        print(f"Orchestrator: Received text input for full_inference_old_placeholder: '{text_input}'") # Renamed

        # DUMMY IMPLEMENTATION:
        # Simulates processing for one layer using a randomly generated input activation.
        # This input activation's shape (1, 100) is chosen to be compatible with the
        # dummy TENSOR_SLICE in shard.py, which is (100, slice_width).
        # (batch_size=1, features_in=100)
        dummy_input_activation = np.random.rand(1, 100).astype(np.float32)

        print(f"Orchestrator: Using DUMMY input activation of shape {dummy_input_activation.shape} for one layer.")

        # Call the distributed matrix multiplication for this dummy layer.
        # Need to provide tensor_name_for_shards if this were to call the updated run_inference_layer_matmul
        layer_output = self.run_inference_layer_matmul(
            dummy_input_activation,
            layer_id="dummy_llm_layer_1",
            tensor_name_for_shards="placeholder_tensor_name" # Add a placeholder name
        )

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
