"""
Scripts package for brain connectivity pipeline
===============================================
Contains executable scripts for running the analysis pipeline.
"""

# Import modules for programmatic access
from . import a1_train_model
from . import a2_apply_to_task
from . import a3_visualize_results

__all__ = [
    'a1_train_model',
    'a2_apply_to_task', 
    'a3_visualize_results'
]

def run_pipeline():
    """Run the complete pipeline programmatically"""
    print("Running complete brain connectivity pipeline...")

    print("\n[1/3] Training model on PIOP-2...")
    a1_train_model.main()

    print("\n[2/3] Applying model to PIOP-1...")
    a2_apply_to_task.main()

    print("\n[3/3] Creating visualizations...")
    a3_visualize_results.main()

    print("\n Pipeline complete!")
