FROM python:3.9-slim

WORKDIR /app

# Copy protobuf generated code first (less likely to change than app code)
COPY ../protos /app/protos

# Copy requirements file
COPY orchestrator/requirements.txt orchestrator/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r orchestrator/requirements.txt

# Copy orchestrator specific files (after pip install to leverage Docker cache better for code changes)
COPY orchestrator/orchestrator.py orchestrator/orchestrator.py
# Copy default config into the image. This can be overridden by mounting a volume at /app/config.
COPY config/shard_config.json /app/config/shard_config.json

ENV PYTHONUNBUFFERED=1
# The orchestrator.py uses a default CONFIG_PATH that points to /app/config/shard_config.json
# when running from /app (its WORKDIR). This can be overridden by Docker env var if needed.
# ENV CONFIG_PATH="/app/config/shard_config.json"

# Set HuggingFace cache directory - can be mapped to a host volume at runtime
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/models
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Note: Explicit creation of cache directory with specific permissions might be needed
# if the user running inside the container doesn't have write access by default,
# or if running with a mounted volume that has restrictive host permissions.
# Often, Hugging Face libraries handle cache directory creation gracefully.
# RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache/huggingface

CMD ["python", "orchestrator/orchestrator.py"]
