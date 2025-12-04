#!/bin/bash

# Script to run the backend locally with MPS support
# This requires the dependencies to be running (e.g., via docker-compose -f docker-compose.mps.yaml up)

# Set environment variables to match docker-compose configuration
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
export CHROMA_COLLECTION_NAME=python_code_chunks
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/service-account-key.json
export GOOGLE_CLOUD_PROJECT=coms-6998-478617
export GOOGLE_CLOUD_LOCATION=us-central1

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: No virtual environment detected. It is recommended to run this in a venv."
fi

echo "Starting backend with MPS support..."
echo "Ensure you have run: docker-compose -f docker-compose.mps.yaml up -d"

# Run uvicorn with reload enabled
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --reload-dir app --reload-exclude '*/repos/*' --timeout-keep-alive 75 --timeout-graceful-shutdown 10
