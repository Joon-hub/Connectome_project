#!/bin/bash
# Bash script to run the brain connectivity pipeline with a specific fold
# This is called by HTCondor with fold number as argument

# Get the fold number from command line argument (NO SPACES around =)
FOLD=$1

# Check if fold argument was provided
if [ -z "$FOLD" ]; then
    echo "ERROR: No fold number provided!"
    echo "Usage: $0 <fold_number>"
    exit 1
fi

echo "=========================================="
echo "Starting Brain Connectivity Pipeline"
echo "Fold: $FOLD"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Check if activation worked
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed  
mkdir -p data/results
mkdir -p logs

# Show Python version for debugging
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Run the pipeline with the specified fold
echo "Running: python scripts/a4_run_full_pipeline.py --fold $FOLD"
python scripts/a4_run_full_pipeline.py --fold $FOLD

# Capture the exit code
EXIT_CODE=$?

echo "=========================================="
echo "Pipeline completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

# Return the exit code
exit $EXIT_CODE