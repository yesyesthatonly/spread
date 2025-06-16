FROM python:3.9-slim

WORKDIR /app

# Copy protobuf generated code first (less likely to change)
COPY ../protos /app/protos

# Copy shard specific files
COPY shard/requirements.txt shard/requirements.txt
COPY shard/shard.py shard/shard.py

# Install dependencies
RUN pip install --no-cache-dir -r shard/requirements.txt

# Environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV SHARD_ID="shard0"
ENV CONFIG_PATH="/app/config/shard_config.json"
# Note: The config file will need to be mounted or copied into the container at this path

EXPOSE 50051 # Default port, will be overridden by config if specified
CMD ["python", "shard/shard.py"]
