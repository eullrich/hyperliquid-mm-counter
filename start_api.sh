#!/bin/bash

# Start the Hyperliquid Anomaly Metrics API

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python"

cd "$SCRIPT_DIR"

echo "Starting Hyperliquid Anomaly Metrics API..."
echo "Dashboard will be available at: http://localhost:8000"
echo ""

$PYTHON_BIN -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
