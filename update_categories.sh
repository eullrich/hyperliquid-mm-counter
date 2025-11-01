#!/bin/bash

# Category Update Script
# Wrapper script to update token categories from Hyperliquid frontend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/category_updater.py"
LOG_FILE="$SCRIPT_DIR/logs/category_update.log"

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

# Log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "Starting category update..."

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log "ERROR: category_updater.py not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run the Python scraper
cd "$SCRIPT_DIR"
if python3 "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then
    log "Category update completed successfully"
    exit 0
else
    log "ERROR: Category update failed"
    exit 1
fi
