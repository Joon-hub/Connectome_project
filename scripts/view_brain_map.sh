#!/bin/bash

# ==============================
# NiLearn brain map visualizer
# ==============================

# Exit on any error
set -e

# Path to your Python virtual environment
VENV_PATH="venv"

# Path to the NIfTI file
NIFTI_FILE="outputs/gender_task_brain.nii.gz"

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

echo "Launching NiLearn interactive brain viewer for ${NIFTI_FILE} ..."

# Run NiLearn viewer
python - <<'EOF'
from nilearn import plotting
view = plotting.view_img("outputs/gender_task_brain.nii.gz", threshold=0.0)
view.open_in_browser()
EOF

echo "Viewer launched successfully ✅"

