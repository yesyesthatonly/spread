FROM python:3.9-slim

WORKDIR /app

# Copy protobuf generated code first
COPY ../protos /app/protos

# Copy orchestrator specific files
COPY orchestrator/requirements.txt orchestrator/requirements.txt
COPY orchestrator/orchestrator.py orchestrator/orchestrator.py

# Copy config file - assumes it's in the context
# Adjust path if config is managed differently
COPY config/shard_config.json /app/config/shard_config.json

# Install dependencies
RUN pip install --no-cache-dir -r orchestrator/requirements.txt

ENV PYTHONUNBUFFERED=1
# The orchestrator.py currently uses a relative path to config based on its own location.
# ENV CONFIG_PATH="/app/config/shard_config.json" # Already handled by default in orchestrator.py

CMD ["python", "orchestrator/orchestrator.py"]
