"""
Run Complete Brain Connectivity Analysis Pipeline
=================================================
This script runs the entire analysis pipeline from start to finish.
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure imports work relative to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_pipeline.utils import print_section
import scripts.train_model as train_model
import scripts.apply_to_task as apply_to_task
import scripts.visualize_results as visualize_results


def main():
    """Main entry point for running the full brain connectivity pipeline."""
    
    print_section("BRAIN CONNECTIVITY ANALYSIS PIPELINE", width=80)
    print(
        """
    This pipeline will:
      1. Train a classifier on PIOP-2 resting-state data
      2. Apply the model to PIOP-1 gender task data
      3. Create error maps and visualizations
      4. Generate comprehensive reports
    """
    )

    # Parse command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Cross-validation fold number (1–5)",
    )
    args = parser.parse_args()
    fold = args.fold

    print_section(f"RUNNING FOLD {fold}/5", width=80)
    start_time = time.time()

    # Step 1: Train model
    print_section("STEP 1/3: TRAINING MODEL", width=80)
    train_model.main(fold=fold)

    # Step 2: Apply to task
    print_section("STEP 2/3: APPLYING TO TASK DATA", width=80)
    apply_to_task.main(fold=fold)

    # Step 3: Visualize
    print_section("STEP 3/3: CREATING VISUALIZATIONS", width=80)
    visualize_results.main()

    # Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print_section("PIPELINE COMPLETE", width=80)
    print(
        f"""
    Total execution time: {minutes}m {seconds}s

    All results saved to: data/results/
    Analysis complete. Check the results directory for all outputs.
    """
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
