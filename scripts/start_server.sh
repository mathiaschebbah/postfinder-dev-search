#!/bin/bash
# Start script for production server on Mac Studio

cd "$(dirname "$0")/.."

# Logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Start uvicorn with production settings
exec uv run uvicorn server_local:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    2>&1 | tee -a "$LOG_DIR/server.log"
