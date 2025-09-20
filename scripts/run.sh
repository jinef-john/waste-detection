#!/bin/bash
# Smart Trash Detection System - Run Script

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the system with passed arguments
python3 main.py "$@"