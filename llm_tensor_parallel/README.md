# LLM Tensor Parallelism with gRPC

This project demonstrates a framework for LLM inference using tensor parallelism, distributing tensor computations (specifically matrix multiplications for MLP layers) across multiple remote shards using gRPC. It now integrates with Hugging Face `transformers` to load and use parts of real models like GPT-2/GPT-2-XL.

## Quickstart (Local Dummy Setup)

This guide will get you running a 2-shard setup locally using the **actual model loading and MLP sharding features** (defaulting to "gpt2" for quicker setup).

1.  **Clone Repository & Navigate to Root:**
    ```bash
    # If you haven't already:
    # git clone <repository_url>
    cd llm_tensor_parallel
    ```

2.  **Set Up Python Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install grpcio grpcio-tools numpy
    # (These cover requirements for orchestrator, shard, and tests for the dummy setup)
    ```

3.  **Generate gRPC Code (if needed):**
    (The repository should include pre-generated files. Run this if you modify `*.proto`.)
    ```bash
    python -m grpc_tools.protoc -I./protos --python_out=./protos --grpc_python_out=./protos ./protos/tensor_parallel.proto
    ```

4.  **Start Shard Servers (2 Terminals):**
    *   **Terminal 1 (Shard 0):**
        ```bash
        export SHARD_ID="shard0"
        # On Windows use: set SHARD_ID="shard0"
        python shard/shard.py
        ```
        *(Wait for "Shard shard0 server started..." message)*
    *   **Terminal 2 (Shard 1):**
        ```bash
        export SHARD_ID="shard1"
        # On Windows use: set SHARD_ID="shard1"
        python shard/shard.py
        ```
        *(Wait for "Shard shard1 server started..." message)*
    *(The default `config/shard_config.json` is used, configuring shard0 on port 50051 and shard1 on port 50052. See 'Configuration for Sharding' section for more details.)*

5.  **Run the Orchestrator (New Terminal):**
    *   **Terminal 3:**
        ```bash
        python orchestrator/orchestrator.py
        ```
        *(Observe output like "Orchestrator: Inference result: Processed output (shape: (1, 100))" indicating success with dummy data.)*

6.  **Run Tests (Optional):**
    *   **Terminal (any of the above, or new):**
        ```bash
        python -m unittest discover tests
        ```

## Architecture

<!-- Diagram: High-level architecture showing Orchestrator, multiple Shards (on different machines/VMs), and gRPC communication flow. -->

-   **Orchestrator**:
    -   Manages the overall inference process.
    -   Loads a model's architecture (conceptually).
    -   Tokenizes input and prepares data for each layer.
    -   For layers involving large weight matrices, it sends the input (or relevant parts of it) to multiple shards.
    -   Gathers processed tensor slices from shards and assembles the layer's output.
    -   Handles communication with shards via gRPC.
-   **Shards**:
    -   gRPC servers, each hosting a slice of one or more large tensors (e.g., columns of a weight matrix).
    -   Each shard loads its tensor slice based on a shared configuration (`config/shard_config.json`).
    -   Exposes a `ComputeMatMul` gRPC method that takes an input tensor, performs matrix multiplication with its local tensor slice, and returns the result slice.
-   **gRPC Interface**:
    -   Defined in `protos/tensor_parallel.proto`.
    -   Specifies the `TensorParallelService` with methods like `ComputeMatMul`.
    -   Uses `Tensor` messages to transmit tensor data (dimensions and values).
-   **Configuration (`config/shard_config.json`)**:
    -   Defines `shard_id`, `ip`, `port` for network addressing.
    -   Includes `model_name` to specify which Hugging Face model the shard should load.
    -   `axis`, `slice_start`, `slice_end` define how each shard slices the tensors listed in its `shardeable_tensor_templates` (currently focused on MLP layers).

## Key Features & Current Scope

*   **Real Model Integration**: Loads actual Hugging Face models (e.g., GPT-2, GPT-2-XL) in both orchestrator and shards.
*   **MLP Layer Sharding**: The MLP layers (`c_fc` and `c_proj` weights) within each transformer block are sharded (column-wise) across the configured shards. The orchestrator distributes the computation for these specific matrix multiplications.
*   **Non-Sharded Operations on Orchestrator**: Embeddings, layer normalizations, residual connections, activation functions (GELU for MLP), and currently the entire attention mechanism (including QKV projection and output projection) are executed on the orchestrator. The final language model head for logits is also on the orchestrator.
*   **Byte-based Tensor Serialization**: Uses `numpy.tobytes()` and `np.frombuffer()` for more efficient tensor data transfer via gRPC.
*   **Basic End-to-End Inference**: Capable of generating text autoregressively for a given prompt.
*   **Limitations**:
    *   **Attention Sharding**: The attention mechanism's linear layers (`c_attn` for QKV, `c_proj` for output) are *not* currently sharded in the `full_inference` pipeline and run on the orchestrator. This is a key area for future sharding enhancement.
    *   **KV Caching**: Not implemented for generation, leading to re-computation for previous tokens in each step.
    *   **Error Handling & Robustness**: Basic; could be improved for production scenarios.
    *   **Configuration Rigidity**: The current shard logic applies the same slicing parameters from `shard_config.json` to all predefined shardeable tensor templates. This requires careful configuration if these tensors have different shardeable dimension sizes.

## Project Structure

```
llm_tensor_parallel/
├── orchestrator/       # Orchestrator code
│   ├── orchestrator.py
│   ├── Dockerfile
│   └── requirements.txt
├── shard/              # Shard server code
│   ├── shard.py
│   ├── Dockerfile
│   └── requirements.txt
├── protos/             # Protobuf definitions and generated Python code
│   ├── tensor_parallel.proto
│   ├── tensor_parallel_pb2.py
│   └── tensor_parallel_pb2_grpc.py
├── config/             # Configuration files
│   └── shard_config.json
├── tests/              # Unit and integration tests
│   ├── __init__.py
│   ├── test_orchestrator.py
│   ├── test_shard.py
│   └── test_integration.py
├── README.md
└── .gitignore
```

## Detailed Setup Instructions

### Step 1: Prerequisites
*   **Python**: Version 3.8 or higher is recommended.
*   **pip**: Python package installer, usually comes with Python.
*   **(Optional) Docker**: If you plan to run the components in containers.

### Step 2: Clone Repository
If you haven't cloned the project repository yet:
```bash
# git clone <repository_url>
# cd llm_tensor_parallel
```
(Assuming you are in the `llm_tensor_parallel` root directory for subsequent steps).

### Step 3: Set Up Python Virtual Environment (Recommended)
Using a virtual environment helps manage project dependencies and avoids conflicts with system-wide Python packages.
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# For Windows: venv\Scripts\activate
```

### Step 4: Install Dependencies
Core dependencies now include `transformers` and `torch`.
```bash
pip install grpcio grpcio-tools numpy transformers torch
# (These cover requirements for orchestrator, shard, and tests)
```
Alternatively, install directly from the requirements files:
```bash
pip install -r orchestrator/requirements.txt
pip install -r shard/requirements.txt
# Ensure grpcio-tools is installed if you need to regenerate protobuf code:
# pip install grpcio-tools
```

### Step 5: Generate gRPC Code
The repository includes pre-generated Python gRPC files from `tensor_parallel.proto`. If you modify the `.proto` file, or if they are missing, regenerate them:

From the `llm_tensor_parallel` root directory, run:
```bash
python -m grpc_tools.protoc -I./protos --python_out=./protos --grpc_python_out=./protos ./protos/tensor_parallel.proto
```
Also, ensure an empty `protos/__init__.py` file exists in the `protos` directory to make the generated modules importable. You can create it if it's missing:
```bash
touch protos/__init__.py # For Linux/macOS
# On Windows, you can create an empty file named __init__.py in the protos folder manually.
```

## Configuration for Sharding (`config/shard_config.json`)

The `config/shard_config.json` file is central to this tensor parallelism framework. It dictates how different parts of a large tensor (from a Hugging Face model like GPT-2) are distributed across various shard servers and how the orchestrator can reach them.

**Structure:**
The file is a JSON array, where each element in the array is an object representing the configuration for a single shard.

```json
[
  {
    "shard_id": "shard0",
    "ip": "localhost",
    "port": 50051,
    "axis": "columns",
    "slice_start": 0,
    "slice_end": 50
  },
  {
    "shard_id": "shard1",
    "ip": "localhost",
    "port": 50052,
    "axis": "columns",
    "slice_start": 50,
    "slice_end": 100
  }
  // ... more shard configurations
]
```

**Key Fields Detailed:**

*   `shard_id` (string): A unique identifier for the shard. This ID is used by the `shard.py` script (via the `SHARD_ID` environment variable) to find its specific configuration details within this JSON file.
*   `model_name` (string, optional): Specifies the Hugging Face model identifier (e.g., `"gpt2"`, `"gpt2-xl"`) that this shard (and the orchestrator) should load. Defaults to `"gpt2-xl"` in the shard if not provided, but it's best to be explicit.
*   `ip` (string): The IP address or hostname that the shard server will bind to. For local testing, `"localhost"` is common.
*   `port` (integer): The network port on which the shard's gRPC server will listen.
*   `axis` (string): Specifies the dimension along which tensors are sliced.
    *   `"columns"`: The primary supported mode. The shard holds a specific range of *columns* of a weight matrix. For `Y = XW`, each shard holds `W_i` where `W = [W_0, W_1, ..., W_n]`.
    *   `"rows"`: The shard holds a specific range of *rows*.
*   `slice_start` (integer): The starting index of the slice this shard is responsible for, along the specified `axis`.
*   `slice_end` (integer): The ending index (exclusive) of the slice. For `axis: "columns"`, this defines columns `slice_start` through `slice_end - 1`.

<!-- Diagram: Illustration of a weight matrix being column-sharded, with Input (X) going to each shard, and partial results (X @ W_slice) being returned and concatenated. -->

**Scaling to N Shards:**
To scale to `N` shards, add `N` objects to the JSON list. Each needs a unique `shard_id`, its network details, and correctly calculated `slice_start`/`slice_end` values for the intended sharded dimension. For example, to shard a tensor with 3072 columns (like GPT-2's `c_fc` output dimension) across 2 shards:
*   Shard 0: `"slice_start": 0`, `"slice_end": 1536`, `"axis": "columns"`
*   Shard 1: `"slice_start": 1536`, `"slice_end": 3072`, `"axis": "columns"`

### Tensor Slicing Notes for Real Models
*   **Universal Slicing Parameters**: The `shard.py` script currently applies the *same* `slice_start`, `slice_end`, and `axis` from its configuration to *all* tensor names defined in its internal `shardeable_tensor_templates` list (e.g., `transformer.h.{}.mlp.c_fc.weight`, `transformer.h.{}.mlp.c_proj.weight`).
*   **Dimension Compatibility**: It's crucial that these slicing parameters are valid for the dimensions of all tensors a shard attempts to slice according to its templates.
    *   For example, in GPT-2, `mlp.c_fc.weight` might be `(n_embd, 4*n_embd)` and `mlp.c_proj.weight` might be `(4*n_embd, n_embd)`. If sharding both column-wise on their *output* dimension (second dimension), `c_fc` needs slices up to `4*n_embd` and `c_proj` up to `n_embd`. A single `slice_end` in `shard_config.json` might not be suitable for both if it's too large for `c_proj` or too small to fully utilize sharding for `c_fc`.
    *   **Recommendation**: For the current implementation, it's best to tailor the `shard_config.json` and the `shardeable_tensor_templates` in `shard.py` to focus on a specific set of tensors that share the same shardeable dimension size (e.g., all `c_fc` layers, or all `c_proj` layers, if sharding them by their output columns). The integration tests, for instance, configure sharding for the `mlp.c_fc` layer's output dimension. Shards might log errors for other templates if the dimensions don't align with the config, but computation will proceed for correctly sliced tensors. Future enhancements could allow per-tensor or per-template slicing rules in the configuration.

## Running the System Locally

**Important: Model Downloads & Cache**
*   Running the system will now download the specified Hugging Face model (e.g., "gpt2" or "gpt2-xl" as defined in `config/shard_config.json` or the orchestrator's default) if it's not already cached by `transformers`. This can take significant time and disk space on the first run.
*   To control the download location, you can set Hugging Face environment variables *before* running the scripts:
    *   `HF_HOME`: Specifies the main directory for Hugging Face caches (e.g., `~/.cache/huggingface`).
    *   `TRANSFORMERS_CACHE`: More specific cache for models.
    *   `HF_DATASETS_CACHE`: For datasets, though not directly used by this project yet.
    Example: `export HF_HOME=/path/to/my/hf_cache`

This section guides you through running the system with actual model loading and sharding of MLP layers.

### Step 1: Understand the Configuration (`config/shard_config.json`)
Review the "Configuration for Sharding (`config/shard_config.json`)" section above for a detailed explanation. Ensure your configuration specifies the desired `model_name` (e.g., "gpt2" for quicker tests, "gpt2-xl" for larger scale) and that `slice_start`/`slice_end`/`axis` are appropriate for the chosen model's tensor dimensions you intend to shard (primarily MLP layers' weights in the current setup).

### Step 2: Start the Shard Servers

Each shard server must be run in its own dedicated terminal window. This allows them to operate independently and listen on their configured ports. The `SHARD_ID` environment variable tells each `shard.py` script which configuration entry to use from the `config/shard_config.json` file.

Navigate to the `llm_tensor_parallel` root directory in each new terminal.

*   **Terminal 1 (For Shard 0):**
    ```bash
    # For Linux/macOS:
    export SHARD_ID="shard0"
    # For Windows (Command Prompt):
    # set SHARD_ID="shard0"
    # For Windows (PowerShell):
    # $env:SHARD_ID="shard0"

    # The shard script uses a default CONFIG_PATH relative to its location:
    # ../config/shard_config.json
    python shard/shard.py
    ```
    *Wait for a message like: `Shard shard0 server started successfully on port 50051.` (The port number comes from your `config/shard_config.json`).*

*   **Terminal 2 (For Shard 1):**
    ```bash
    # For Linux/macOS:
    export SHARD_ID="shard1"
    # For Windows (Command Prompt):
    # set SHARD_ID="shard1"
    # For Windows (PowerShell):
    # $env:SHARD_ID="shard1"

    python shard/shard.py
    ```
    *Wait for a message like: `Shard shard1 server started successfully on port 50052.` (The port number comes from your `config/shard_config.json`).*

*   **For Additional Shards (e.g., Shard N):**
    Open a new terminal for each additional shard. Set the `SHARD_ID` environment variable accordingly (e.g., `shard2`, `shard3`, etc.) and then run `python shard/shard.py`. Ensure that a corresponding configuration for each new `SHARD_ID` exists in your `config/shard_config.json` file.

It's crucial that each shard server is running and has outputted its "started successfully" message before you proceed to start the orchestrator. The orchestrator relies on these shards being operational to distribute work.

### Step 3: Run the Orchestrator
Once the shard servers are running, you can start the orchestrator.

Open a new terminal and navigate to the `llm_tensor_parallel` root directory.

```bash
# The orchestrator script also uses a default CONFIG_PATH relative to its location:
# ../config/shard_config.json
python orchestrator/orchestrator.py
```
The orchestrator will:
1.  Load the `config/shard_config.json`.
2.  Attempt to connect to the shard servers specified.
3.  (After a brief wait) Send a dummy input for a distributed matrix multiplication.
4.  Print the shape of the received (concatenated) result. You should see something like:
    `Orchestrator: Generated text: 'Hello world this is a test...'` (or similar generated text).

This completes a local run of the system with real model processing.

## Running with Docker (Optional)

Dockerfiles are provided for both the orchestrator (`orchestrator/Dockerfile`) and shard (`shard/Dockerfile`). These allow you to run the components in isolated containers. Remember to consider model download locations and caching when using Docker.

**Model Cache Volume (Recommended):**
To persist Hugging Face model downloads across container runs and share the cache between the host and containers (or among multiple containers), mount a host directory to the container's cache location. The Dockerfiles are already configured to use `/app/.cache/huggingface` via `HF_HOME`.
Example:
```bash
# On your host machine, create a directory for the cache:
# mkdir -p ./my_hf_cache
# (Ensure this path is absolute when using -v for clarity, or use $(pwd)/my_hf_cache)

# Then, when running containers, add the volume mount:
# -v /path/to/your/host/my_hf_cache:/app/.cache/huggingface
```
Ensure the `HF_HOME`, `TRANSFORMERS_CACHE`, and `HF_DATASETS_CACHE` environment variables (set in the Dockerfiles) point to this mounted path or subdirectories within it.

### 1. Build the Docker Images
From the `llm_tensor_parallel` root directory:
```bash
docker build -t llm_shard -f shard/Dockerfile .
docker build -t llm_orchestrator -f orchestrator/Dockerfile .
```

### 2. Understanding Network Configuration for Docker
Networking between Docker containers, or between a container and the host machine, requires careful configuration of IP addresses or hostnames in `config/shard_config.json`. The default `localhost` will not work as expected for inter-container communication.

*   **Option A: All Containers on a Custom Docker Network (Recommended for Multi-Container Apps)**
    This is the most straightforward approach when running both the orchestrator and shards as Docker containers.
    1.  **Create a Docker Network:**
        ```bash
        docker network create llm_network
        ```
    2.  **Modify `shard_config.json` for Service Names:**
        You'll need a version of `shard_config.json` where the `ip` field for each shard points to its **container name** (which will also be its service name on the Docker network).
        For example, create `config/docker_shard_config.json`:
        ```json
        // Example: config/docker_shard_config.json
        [
          {
            "shard_id": "shard0",
            "ip": "shard0", // <-- Service name for shard0
            "port": 50051,
            "axis": "columns",
            "slice_start": 0,
            "slice_end": 50
          },
          {
            "shard_id": "shard1",
            "ip": "shard1", // <-- Service name for shard1
            "port": 50052,
            "axis": "columns",
            "slice_start": 50,
            "slice_end": 100
          }
        ]
        ```
    3.  **Ensure Docker Images Use the Correct Config:**
        *   **Option 1 (Build-time):** Modify your `orchestrator/Dockerfile` and `shard/Dockerfile` to copy this `config/docker_shard_config.json` to `/app/config/shard_config.json` inside the image. This is suitable if the Docker-specific config is static.
        *   **Option 2 (Run-time Mount):** Keep the default `shard_config.json` in the image (which uses `localhost`) and mount your Docker-specific configuration file when running the container. This is more flexible. For example:
            `-v $(pwd)/config/docker_shard_config.json:/app/config/shard_config.json`

*   **Option B: Orchestrator on Host, Shards in Docker (or other mixed setups)**
    1.  **Port Mapping for Shards:** When running shard containers, map their gRPC ports to ports on your host machine.
        Example for a shard container: `-p 50051:50051` (maps host port 50051 to container port 50051).
    2.  **Configure Orchestrator:** The `shard_config.json` used by the orchestrator (running on the host) must then use the host's IP address (e.g., `192.168.x.x` or `host.docker.internal` if your Docker version supports it for host access from containers) and the *host ports* you mapped.

### 3. Running Shard Containers (Example with Custom Network)
This example assumes you are using Option A (custom network), a Docker-aware `shard_config.json` (using service names like "shard0", "shard1" as IPs), and that this config is either copied into your images or mounted. Remember to include the volume mount for the cache.

*   **Shard 0:**
    ```bash
    docker run -d --rm --name shard0 --network llm_network \
      -v /path/to/host/my_hf_cache:/app/.cache/huggingface \
      -e SHARD_ID="shard0" \
      -e CONFIG_PATH="/app/config/shard_config.json" \
      -e HF_HOME="/app/.cache/huggingface" \
      llm_shard
    ```
*   **Shard 1:**
    ```bash
    docker run -d --rm --name shard1 --network llm_network \
      -v /path/to/host/my_hf_cache:/app/.cache/huggingface \
      -e SHARD_ID="shard1" \
      -e CONFIG_PATH="/app/config/shard_config.json" \
      -e HF_HOME="/app/.cache/huggingface" \
      llm_shard
    ```
    Docker's internal DNS will resolve `shard0` and `shard1` (if used as IPs in config) within `llm_network`.

### 4. Running Orchestrator Container (Example with Custom Network)
The orchestrator joins the same network.
```bash
docker run --rm --network llm_network \
  -v /path/to/host/my_hf_cache:/app/.cache/huggingface \
  -e CONFIG_PATH="/app/config/shard_config.json" \
  # If mounting a specific Docker config: -v $(pwd)/config/docker_shard_config.json:/app/config/shard_config.json
  -e HF_HOME="/app/.cache/huggingface" \
  llm_orchestrator
```
The orchestrator should connect to shards using their service names (e.g., `shard0:50051`).

**Important Notes for Docker:**
*   **Configuration File:** The `CONFIG_PATH` inside containers is `/app/config/shard_config.json`. Ensure the correct (Docker-network-aware) version of this file is present at that path, either by being `COPY`ed in the Dockerfile or mounted with `-v`.
*   **Cache Consistency**: Using the same mounted cache volume for all containers (orchestrator and shards) ensures models are downloaded only once.
*   **Logs**: Always check container logs (`docker logs <container_name>`) for issues.

## Running Tests

From the `llm_tensor_parallel` root directory:

```bash
# Ensure test dependencies are installed (transformers, torch, etc.)
# pip install -r orchestrator/requirements.txt # (or a combined test_requirements.txt)
python -m unittest discover tests
```
This will run all unit and integration tests. Integration tests now use the "gpt2" model by default and will download it if not cached (respecting `HF_HOME` if set in the test execution environment).

## Troubleshooting / FAQ

Here are some common issues and solutions:

1.  **Python Import Errors (e.g., `ModuleNotFoundError: No module named 'protos'` or `'shard'`)**
    *   **Virtual Environment**: Ensure your virtual environment is activated.
    *   **PYTHONPATH**: If running scripts directly from subdirectories, Python might not find the modules.
        *   It's often best to run Python commands from the project root directory (`llm_tensor_parallel/`). For example: `python shard/shard.py` or `python -m tests.test_integration`.
        *   The test scripts include `sys.path` modifications as a workaround, but for general execution, running from the root is preferred.
        *   Consider setting `PYTHONPATH` if you consistently run from other locations: `export PYTHONPATH=$PYTHONPATH:/path/to/your/llm_tensor_parallel`.
    *   **Generated gRPC Code**: Make sure you have run the gRPC code generation step if you modified `tensor_parallel.proto` or if the `protos/tensor_parallel_pb2.py` and `protos/tensor_parallel_pb2_grpc.py` files are missing. Also ensure `protos/__init__.py` exists.

2.  **gRPC Connection Issues (Orchestrator can't connect to Shards)**
    *   **Shard Not Running**: Double-check that all shard servers were started successfully in their separate terminals and that you saw their startup confirmation messages (e.g., "Shard shardX server started...").
    *   **Incorrect IP/Port**: Verify that the `ip` and `port` in your `config/shard_config.json` match exactly where your shards are listening. Remember, the orchestrator reads this file to find the shards.
    *   **Firewall**: A local or network firewall could be blocking connections to the shard ports.
    *   **`SHARD_ID` Not Set**: If a shard server doesn't seem to start on the expected port, ensure the `SHARD_ID` environment variable was correctly set in the terminal where that shard is running. The `shard.py` script uses this ID to find its configuration (including port) in `shard_config.json`.
    *   **gRPC Server/Client Mismatch**: Less common with this setup, but ensure both orchestrator and shards are using the same compiled `_pb2.py` and `_pb2_grpc.py` files. Regenerate them if in doubt.

3.  **Matrix Multiplication Errors in Shard (e.g., `ValueError: shapes (X,Y) and (A,B) not aligned`)**
    *   This usually means the input tensor received by the shard from the orchestrator has dimensions incompatible with the shard's local `TENSOR_SLICE`.
    *   **Check Dummy Tensor Logic**: Review the "Expectations for the Current Dummy Tensor Implementation" subsection in the "Configuration for Sharding" section. The orchestrator's dummy input must align with the shard's dummy tensor (e.g., if shard slice is `(100, slice_width)`, orchestrator input `(batch, features_in)` must have `features_in=100`).
    *   **Logging**: Add print statements in `shard.py` (input tensor shape, slice shape) and `orchestrator.py` (tensor shape before sending) to debug.

4.  **Docker Networking Issues**
    *   **`localhost` in Config**: If you're running shards in Docker containers and the orchestrator (either also in Docker or on the host) is trying to connect, `localhost` in `shard_config.json` usually won't work as expected. `localhost` inside a container refers to the container itself.
        *   **Docker Network & Service Names**: If all containers (orchestrator and shards) are on the same `docker network create ...` custom network, you can often use the container names (e.g., `shard0`, `shard1`) as hostnames in `shard_config.json` (assuming the orchestrator's Dockerfile copies this modified config).
        *   **Host IP / Port Mapping**: If the orchestrator is running on the host or outside the Docker network of the shards, you'll need to map shard container ports to host ports (e.g., `docker run -p 50051:50051 ...`) and use the host's IP (or `host.docker.internal` on some systems for the host from within a container) in `shard_config.json`.
    *   **Firewall (again)**: Ensure Docker isn't being blocked by a firewall.

5.  **Tests Failing (Especially Integration Tests)**
    *   **Shard Startup Time**: `test_integration.py` uses `time.sleep()` to wait for shards. If your system is slow, this might not be enough. You might see errors about shards not being ready. Increase the sleep duration in `setUpClass` as a temporary fix.
    *   **Environment**: Ensure tests are run from the project root or that `PYTHONPATH` is correctly configured so they can find all modules. `python -m unittest discover tests` from the root is generally reliable.
    *   **Port Conflicts**: If other applications are using the ports defined for tests (e.g., 50080, 50081), the integration tests might fail to start shards.

## Future Enhancements

*   **Real Model Loading**: Integrate with Hugging Face `transformers` to load actual LLM weights (e.g., GPT-2 XL).
*   **Automatic Tensor Slicing**: Implement logic to automatically slice tensors from a loaded model based on the shard configuration.
*   **Advanced Tensor Parallelism**: Support more complex tensor parallelism strategies (e.g., row parallelism for other types of layers, pipeline parallelism).
*   **Efficient Tensor Serialization**: Use `bytes` for tensor data in protobuf messages for better performance with large tensors (e.g., `tensor.tobytes()`).
*   **Robust Error Handling and Shard Management**: Implement better shard failure detection, retries, and potentially dynamic shard registration/discovery.
*   **Asynchronous Orchestrator**: Make the orchestrator fully asynchronous for better performance.
*   **Benchmarking**: Add tools for benchmarking performance.
