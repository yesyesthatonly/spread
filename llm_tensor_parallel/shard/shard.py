import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import grpc
from concurrent import futures
import time
import numpy as np
import json
import os

# Import Hugging Face libraries
from transformers import AutoModelForCausalLM
import torch

# Import generated gRPC classes
from protos import tensor_parallel_pb2
from protos import tensor_parallel_pb2_grpc

# SHARD_ID and CONFIG_PATH are now read within the Servicer's initialization
# SHARD_ID = os.environ.get("SHARD_ID", "shard0")
# CONFIG_PATH = os.environ.get("CONFIG_PATH", "../config/shard_config.json")

class TensorParallelServiceServicer(tensor_parallel_pb2_grpc.TensorParallelServiceServicer):
    """
    gRPC service implementation for tensor parallelism.
    This class handles incoming RPC calls defined in tensor_parallel.proto.
    It now loads specified layers of a Hugging Face model and serves slices of them.
    """
    def __init__(self):
        """
        Initializes the servicer, loading the model and tensor slices.
        """
        self.tensor_slices_cache = {} # Cache for storing tensor slices
        self.model = None             # Will hold the loaded Hugging Face model
        self.shard_id = None          # Will be set from ENV
        self.shard_config = None      # Will be loaded from config file
        self._initialize_shard()      # Load model and tensor slices

    def _initialize_shard(self):
        """
        Loads the shard's configuration, the specified Hugging Face model,
        and extracts the necessary tensor slices based on the configuration.
        """
        self.shard_id = os.environ.get("SHARD_ID", "shard0")
        config_path = os.environ.get("CONFIG_PATH", "../config/shard_config.json")

        print(f"Shard {self.shard_id}: Initializing...")
        try:
            with open(config_path, 'r') as f:
                all_configs = json.load(f)
            self.shard_config = next((s_conf for s_conf in all_configs if s_conf["shard_id"] == self.shard_id), None)

            if not self.shard_config:
                error_msg = f"Configuration for shard {self.shard_id} not found in {config_path}."
                print(f"Shard {self.shard_id}: {error_msg}")
                raise ValueError(error_msg)
        except Exception as e:
            print(f"Shard {self.shard_id}: Error loading shard configuration from {config_path}: {e}")
            raise # Critical failure if config cannot be loaded

        # Get model_name from shard_config, defaulting to "gpt2-xl"
        # This allows different sets of shards to potentially serve different models or versions.
        model_name = self.shard_config.get("model_name", "gpt2-xl")
        print(f"Shard {self.shard_id}: Loading model '{model_name}' for tensor slicing.")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval() # Set model to evaluation mode (disables dropout, etc.)
            print(f"Shard {self.shard_id}: Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Shard {self.shard_id}: Error loading model '{model_name}': {e}")
            raise # Critical failure if model cannot be loaded

        self._load_all_tensor_slices()

    def _load_all_tensor_slices(self):
        """
        Extracts and caches tensor slices from the loaded model based on shard_config.
        Populates `self.tensor_slices_cache`.
        """
        print(f"Shard {self.shard_id}: Preparing to load tensor slices...")
        if not self.model:
            print(f"Shard {self.shard_id}: Model not loaded. Cannot load slices.")
            # This should ideally not happen if _initialize_shard succeeded.
            return

        slice_start = self.shard_config["slice_start"]
        slice_end = self.shard_config["slice_end"]
        axis = self.shard_config["axis"] # Expected "columns" or "rows"

        # Determine the number of layers from the model's configuration
        # (e.g., for GPT-2, this is `n_layer`)
        num_layers = self.model.config.n_layer

        # Define templates for names of tensors that are typically sharded in GPT-like models.
        # These templates correspond to weights in attention blocks and MLP blocks.
        shardeable_tensor_templates = [
            "transformer.h.{}.attn.c_attn.weight",  # Attention (Query, Key, Value projection) weights
            # "transformer.h.{}.attn.c_attn.bias",  # Corresponding biases (can also be sharded)
            "transformer.h.{}.attn.c_proj.weight", # Attention output projection weights
            "transformer.h.{}.mlp.c_fc.weight",    # MLP feed-forward layer weights
            "transformer.h.{}.mlp.c_proj.weight",   # MLP projection layer weights
        ]

        # In a more comprehensive setup, one might also shard:
        # - "transformer.wte.weight" (Word Token Embeddings)
        # - "lm_head.weight" (Language Model Head - often tied/shared with wte.weight)
        # Sharding these would depend on vocabulary size and chosen sharding strategy (e.g., row-wise).

        named_parameters = dict(self.model.named_parameters())

        for i in range(num_layers): # Iterate through each layer of the model
            for template in shardeable_tensor_templates:
                tensor_name = template.format(i) # Construct specific tensor name for the layer

                if tensor_name in named_parameters:
                    full_tensor = named_parameters[tensor_name].data # Get the tensor data

                    # Detach tensor from computation graph and move to CPU for slicing.
                    # This ensures we are working with a plain tensor and not affecting gradients.
                    full_tensor = full_tensor.detach().cpu()

                    sliced_tensor_part = None
                    if axis == "columns":
                        # Column-wise sharding: tensor is sliced along its second dimension (columns).
                        # Example: For a weight matrix (K, N), a slice is (K, N_slice).
                        sliced_tensor_part = full_tensor[:, slice_start:slice_end]
                    elif axis == "rows":
                        # Row-wise sharding: tensor is sliced along its first dimension (rows).
                        # Example: For a weight matrix (K, N), a slice is (K_slice, N).
                        sliced_tensor_part = full_tensor[slice_start:slice_end, :]
                    else:
                        print(f"Shard {self.shard_id}: Unknown axis '{axis}' for tensor {tensor_name}. Skipping slicing for this tensor.")
                        continue # Skip if axis configuration is not recognized

                    # Store the slice as a NumPy array in the cache.
                    # NumPy is used here for consistency with the existing MatMul implementation,
                    # but slices could also be kept as PyTorch tensors.
                    self.tensor_slices_cache[tensor_name] = sliced_tensor_part.numpy()
                    print(f"Shard {self.shard_id}: Loaded slice for {tensor_name} with shape {self.tensor_slices_cache[tensor_name].shape}")
                else:
                    print(f"Shard {self.shard_id}: Tensor {tensor_name} not found in the loaded model. Skipping.")

        # Example for sharding embeddings (if configured, not fully implemented here)
        # if self.shard_config.get("shard_embeddings", False):
        #     wte_name = "transformer.wte.weight"
        #     if wte_name in named_parameters:
        #         # Add logic for wte sharding, likely row-wise based on vocab size distribution
        #         pass

        if not self.tensor_slices_cache:
            print(f"Shard {self.shard_id}: WARNING - No tensor slices were loaded into cache. "
                  f"Check shard configuration (slice_start/end, axis), model structure ('{model_name}'), "
                  f"and shardeable_tensor_templates list in shard.py.")
        else:
            print(f"Shard {self.shard_id}: Finished loading {len(self.tensor_slices_cache)} tensor slices into cache.")


    def ComputeMatMul(self, request, context):
        """
        Computes a matrix multiplication using a specific tensor slice requested by the client.
        """
        requested_tensor_name = request.tensor_name # Name of the tensor slice to use (e.g., "transformer.h.0.attn.c_attn.weight")

        print(f"Shard {self.shard_id}: Received ComputeMatMul request for tensor '{requested_tensor_name}'.")

        # Check if the requested tensor slice is in this shard's cache
        if not requested_tensor_name or requested_tensor_name not in self.tensor_slices_cache:
            error_msg = f"Tensor slice for '{requested_tensor_name}' not found on shard {self.shard_id}. Available slices: {list(self.tensor_slices_cache.keys())}"
            print(f"Shard {self.shard_id}: {error_msg}")
            context.abort(grpc.StatusCode.NOT_FOUND, error_msg)
            return tensor_parallel_pb2.ComputeResponse()

        current_tensor_slice = self.tensor_slices_cache[requested_tensor_name]

        input_tensor_proto = request.input_tensor
        input_dims = list(input_tensor_proto.dims)
        input_dtype = np.dtype(input_tensor_proto.dtype) # Get dtype
        try:
            # Deserialize input tensor from protobuf message to NumPy array
            input_data = np.frombuffer(input_tensor_proto.serialized_data, dtype=input_dtype).reshape(input_dims)
        except ValueError as e:
            error_msg = f"Error deserializing input tensor on shard {self.shard_id} for tensor {requested_tensor_name}: {e}. Dims: {input_dims}, dtype: {input_dtype}, expected size from dims: {np.prod(input_dims) * input_dtype.itemsize}, actual_data_len: {len(input_tensor_proto.serialized_data)}"
            print(error_msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            return tensor_parallel_pb2.ComputeResponse()

        print(f"Shard {self.shard_id}: Input tensor shape: {input_data.shape}, Slice for '{requested_tensor_name}' shape: {current_tensor_slice.shape}")

        # Perform matrix multiplication: result = input_data @ current_tensor_slice
        try:
            result_data = np.matmul(input_data, current_tensor_slice)
        except ValueError as e:
            error_msg = (f"Matrix multiplication error on shard {self.shard_id} for tensor '{requested_tensor_name}': {e}. "
                         f"Input shape {input_data.shape}, Slice shape {current_tensor_slice.shape}")
            print(error_msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            return tensor_parallel_pb2.ComputeResponse()

        print(f"Shard {self.shard_id}: Computation for '{requested_tensor_name}' complete. Result shape: {result_data.shape}")

        # Serialize output tensor slice from NumPy array to protobuf message
        output_tensor_proto = tensor_parallel_pb2.Tensor(
            dims=list(result_data.shape),
            serialized_data=result_data.tobytes(), # Use tobytes()
            dtype=str(result_data.dtype)           # Store dtype as string
        )
        return tensor_parallel_pb2.ComputeResponse(output_tensor=output_tensor_proto)

def serve():
    """
    Starts the gRPC server for this shard.
    The servicer handles its own initialization including model and slice loading.
    """
    # Instantiate the servicer. Initialization (_initialize_shard) is called within its __init__.
    # This will load the model and tensor slices based on ENV vars SHARD_ID and CONFIG_PATH.
    try:
        servicer = TensorParallelServiceServicer()
    except Exception as e:
        # If servicer initialization fails (e.g., model loading, config error),
        # the shard cannot start. Log the error and exit.
        # The specific error should have been printed by _initialize_shard.
        print(f"CRITICAL: Shard failed to initialize servicer: {e}. Exiting.")
        return # Exit if servicer setup fails

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tensor_parallel_pb2_grpc.add_TensorParallelServiceServicer_to_server(
        servicer, server
    )

    # Get port from the servicer's loaded shard_config.
    # Use a default if somehow not set, though _initialize_shard should ensure shard_config.
    port_str = "50051" # Default port
    if servicer.shard_config and "port" in servicer.shard_config:
        port_str = str(servicer.shard_config["port"])
    else:
        print(f"Warning: Shard {servicer.shard_id} - port not found in config, using default {port_str}. This might be an issue if shard_config was not loaded correctly.")

    server.add_insecure_port(f"[::]:{port_str}")
    print(f"Shard {servicer.shard_id} server starting on port {port_str}...")
    server.start()
    print(f"Shard {servicer.shard_id} server started successfully on port {port_str}. Awaiting requests.")

    try:
        while True:
            time.sleep(86400)  # Keep main thread alive
    except KeyboardInterrupt:
        print(f"Shard {servicer.shard_id} server stopping due to keyboard interrupt...")
        server.stop(0) # Graceful shutdown
        print(f"Shard {servicer.shard_id} server stopped.")

if __name__ == "__main__":
    serve()
