FROM python:3.9-slim

WORKDIR /app

# Copy protobuf generated code first (less likely to change than app code)
COPY ../protos /app/protos

# Copy requirements file
COPY shard/requirements.txt shard/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r shard/requirements.txt

# Copy shard specific files (after pip install to leverage Docker cache better for code changes)
COPY shard/shard.py shard/shard.py
# Note: The shard.py script determines its config path via CONFIG_PATH env var.
# The actual config/shard_config.json is NOT copied by default into the shard image here,
# as it's expected to be provided via a mounted volume or be present in a base image
# if not running alongside an orchestrator that manages config distribution.
# However, the orchestrator Dockerfile DOES copy a default config.

# Environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV SHARD_ID="shard0" # Default SHARD_ID, should be overridden for each shard instance
ENV CONFIG_PATH="/app/config/shard_config.json" # Assumes config will be available at this path in container

# Set HuggingFace cache directory - can be mapped to a host volume at runtime
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/models
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Note: Explicit creation of cache directory with specific permissions might be needed.
# See comments in orchestrator/Dockerfile.
# RUN mkdir -p /app/cache/huggingface && chmod -R 777 /app/cache/huggingface

# EXPOSE the default port. The actual port used is determined by shard_config.json and SHARD_ID.
EXPOSE 50051

CMD ["python", "shard/shard.py"]
