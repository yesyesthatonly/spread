# LLM Tensor Parallelism with gRPC

This project demonstrates a basic framework for large language model (LLM) inference using tensor parallelism. It distributes tensor computations (specifically matrix multiplications) across multiple remote shards using gRPC.

## Quickstart (Local Dummy Setup)

This guide will get you running a 2-shard setup locally using the dummy tensor logic.

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
-   **Configuration**:
    -   `config/shard_config.json` defines the properties of each shard:
        -   `shard_id`: Unique identifier for the shard.
        -   `ip`: IP address of the shard.
        -   `port`: Port on which the shard's gRPC server listens.
        -   `slice_start`, `slice_end`: Defines the portion of the tensor this shard is responsible for.
        -   `axis`: The axis along which the tensor is sliced (e.g., "columns", "rows").

## Current Status & Limitations (Dummy Implementation)

*   **Dummy Tensor Logic**: The current implementation uses *dummy* tensor data within the shards (`shard/shard.py`). Instead of loading slices from a real LLM (like GPT-2-XL), each shard initializes a random tensor slice. This allows testing the gRPC communication, data flow, and parallel computation logic without the overhead of actual model loading.
*   **Matrix Multiplication Focus**: The primary parallelized operation is matrix multiplication. Other LLM layer operations (activations, normalization) are assumed to be handled by the orchestrator locally in the current simplified model.
*   **Basic Error Handling**: Error handling is basic.
*   **No Real Model Loading**: Full model loading (e.g., from Hugging Face Transformers) and automatic tensor slicing based on a model's architecture are not yet implemented. The orchestrator and shards currently operate on the abstract idea of tensor slices defined purely by the `shard_config.json` and the dummy data generation in `shard.py`.

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
The quickest way to install core dependencies for the dummy setup (orchestrator, shard, and tests) is:
```bash
pip install grpcio grpcio-tools numpy
```
Alternatively, to install from requirements files (currently very similar to the above for the dummy setup):
```bash
pip install -r orchestrator/requirements.txt
pip install -r shard/requirements.txt
# grpcio-tools is listed separately as it's primarily for gRPC code generation.
# numpy is used by both orchestrator and shard, and for tests.
```
For development or if you need to regenerate gRPC code, ensure `grpcio-tools` is installed:
```bash
pip install grpcio-tools
```

### Step 5: Generate gRPC Code
The repository should include pre-generated Python gRPC files in the `protos/` directory (`tensor_parallel_pb2.py`, `tensor_parallel_pb2_grpc.py`).
However, if you modify `protos/tensor_parallel.proto` or if these files are missing, you'll need to regenerate them.

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

The `config/shard_config.json` file is central to this tensor parallelism framework. It dictates how different parts of a large tensor (conceptually, a model's weight matrix) are distributed across various shard servers and how the orchestrator can reach them.

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
*   `ip` (string): The IP address or hostname that the shard server will bind to. For local testing, `"localhost"` is common. In a distributed or containerized setup, this would be the shard's reachable IP address or service name.
*   `port` (integer): The network port on which the shard's gRPC server will listen for incoming requests from the orchestrator.
*   `axis` (string): Specifies the dimension along which the conceptual "full tensor" is sliced.
    *   `"columns"`: This is the primary mode supported by the current dummy implementation. It implies that the shard holds a specific range of *columns* of a larger matrix. For a weight matrix `W` in a layer `Y = XW`, each shard holds `W_i` where `W = [W_0, W_1, ..., W_n]`.
    *   `"rows"`: While not fully demonstrated in the dummy examples for matrix multiplication (which focuses on column sharding of weights), this would imply that the shard holds a specific range of *rows*. This could be useful for other types of operations or different tensor parallelism strategies.
*   `slice_start` (integer): The starting index of the slice this shard is responsible for, along the specified `axis`.
*   `slice_end` (integer): The ending index (exclusive) of the slice this shard is responsible for. For example, if `axis` is `"columns"`, `slice_start` is 0, and `slice_end` is 50, the shard manages columns 0 through 49.

<!-- Diagram: Illustration of a weight matrix being column-sharded, with Input (X) going to each shard, and partial results (X @ W_slice) being returned and concatenated. -->

**Scaling to N Shards:**
To scale your distributed computation to `N` shards, you would simply add `N` objects to the JSON list in `config/shard_config.json`. Each object must have:
1.  A unique `shard_id`.
2.  The correct network details (`ip` and `port`) for that shard.
3.  Carefully calculated `slice_start` and `slice_end` values to ensure the entire tensor is covered without overlaps or gaps. For example, if a weight matrix has 2048 columns and you want to distribute it across 4 shards using column-wise sharding:
    *   Shard 0: `slice_start: 0`, `slice_end: 512`
    *   Shard 1: `slice_start: 512`, `slice_end: 1024`
    *   Shard 2: `slice_start: 1024`, `slice_end: 1536`
    *   Shard 3: `slice_start: 1536`, `slice_end: 2048`
    The `axis` for all these would be `"columns"`.

### Expectations for the Current Dummy Tensor Implementation
It's crucial to understand how the current `shard/shard.py` (specifically its `load_tensor_slice` function) interacts with this configuration, as it uses *dummy data* rather than loading parts of a real model:

*   **Hardcoded Row Dimension (for Column Sharding):** When `axis: "columns"`, the dummy tensor slice created by `shard.py` is hardcoded to have **100 rows**.
*   **Column Dimension from Config:** The number of columns for this dummy slice is determined by `slice_end - slice_start` from its configuration entry in `shard_config.json`. For instance, if a shard has `axis: "columns"`, `slice_start: 0`, and `slice_end: 50`, its dummy tensor slice will be of shape `(100, 50)`.
*   **Orchestrator Compatibility:** The orchestrator's input tensor (especially in the `full_inference` dummy example or test cases) must be mathematically compatible for multiplication with these shard slices. If each shard's slice is `(100, slice_width_i)` due to column sharding, the input activation from the orchestrator must have 100 columns (features). For example, an input of shape `(batch_size, 100)` would be compatible. This is why the default orchestrator examples work with the default shard configurations. If you change the slice configurations or the hardcoded dimension in `shard.py`, you might need to adjust the orchestrator's input accordingly.

**Suitability for Real Models (Briefly):**
For real LLMs, column-wise sharding (where `axis: "columns"` and slices define column ranges) is a common strategy for distributing large weight matrices in feed-forward neural network layers or the dense projection layers within attention mechanisms. This is particularly effective when the inner dimension of the matrix multiplication (often denoted `K` in `A(M,K) @ W(K,N)`) is large and consistent for the input activations (`A`) and the weight matrix slices (`W_i`).

## Running the System Locally (Dummy Tensor Example)

This section guides you through running the system with its current dummy tensor logic. The orchestrator and shards will communicate, but tensor computations will use randomly generated data on the shards.

### Step 1: Understand the Configuration (`config/shard_config.json`)
The `config/shard_config.json` file is crucial for defining how the tensor processing is distributed. (See the "Configuration for Sharding (`config/shard_config.json`)" section above for a detailed explanation).

The default configuration in `config/shard_config.json` sets up two shards on `localhost` (ports `50051` and `50052`) for column-wise sharding. Based on the dummy tensor logic detailed in the configuration section, each shard will effectively load a tensor of shape `(100, 50)`.

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
    `Orchestrator: Inference result: Processed output (shape: (1, 100))`
    (The exact output dimension depends on the `slice_end` of the last shard in the config for the dummy data).

This completes a local run of the system with dummy tensor processing.

## Running with Docker (Optional)

Dockerfiles are provided for both the orchestrator (`orchestrator/Dockerfile`) and shard (`shard/Dockerfile`). These allow you to run the components in isolated containers.

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
This example assumes you are using Option A with a `config/docker_shard_config.json` that uses service names, and this config is correctly placed in `/app/config/shard_config.json` inside the images (either by Dockerfile copy or mount).

*   **Shard 0:**
    ```bash
    docker run -d --rm --name shard0 --network llm_network \
      -e SHARD_ID="shard0" \
      -e CONFIG_PATH="/app/config/shard_config.json" \
      llm_shard
    ```
*   **Shard 1:**
    ```bash
    docker run -d --rm --name shard1 --network llm_network \
      -e SHARD_ID="shard1" \
      -e CONFIG_PATH="/app/config/shard_config.json" \
      llm_shard
    ```
    Docker's internal DNS will resolve `shard0` and `shard1` to their respective container IPs within the `llm_network`.

### 4. Running Orchestrator Container (Example with Custom Network)
The orchestrator also joins the same network and uses the Docker-aware `shard_config.json`.

```bash
docker run --rm --network llm_network \
  -e CONFIG_PATH="/app/config/shard_config.json" \
  # If mounting config: -v $(pwd)/config/docker_shard_config.json:/app/config/shard_config.json
  llm_orchestrator
```
The orchestrator should now be able to connect to `shard0:50051` and `shard1:50052`.

**Important Notes for Docker:**
*   The `CONFIG_PATH` environment variable inside the containers is set to `/app/config/shard_config.json` as per the Dockerfiles. Ensure the correct version of the config file is present at this path inside the running containers.
*   Always check container logs (`docker logs <container_name>`) if you encounter connection issues.

## Running Tests

From the `llm_tensor_parallel` root directory:

```bash
# Ensure test dependencies are installed (e.g. from a requirements_dev.txt or manually)
# pip install ... (numpy, grpcio are needed for tests)
python -m unittest discover tests
```
This will run all unit and integration tests. Integration tests will start and stop shard subprocesses automatically.

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
