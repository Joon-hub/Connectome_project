#!/bin/bash
#
# Wrapper script for hyperparameter tuning job
# Usage: ./run_hyperparameter_tuning.sh <job_id>
#

# Exit on error
set -e

# Get job ID from command line
JOB_ID=$1

echo "========================================"
echo "Hyperparameter Tuning Job ${JOB_ID}"
echo "========================================"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "========================================"

# Activate virtual environment (adjust path as needed)
source /home/sjoon/projects/Connectome_project/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs/tuning

# Run the hyperparameter tuning script
python scripts/hyperparameter_tuning.py --job-id ${JOB_ID}

EXIT_CODE=$?

echo "========================================"
echo "Job ${JOB_ID} completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "========================================"

exit ${EXIT_CODE}