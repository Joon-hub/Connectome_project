#!/bin/bash
# Activate your virtual environment
source venv/bin/activate

# Move into project directory
cd "$(dirname "$0")"

# Default fold to 1 if not provided 
fold=${1:-1}

# Run pipeline script
python scripts/run_full_pipeline.py --fold "$fold"
