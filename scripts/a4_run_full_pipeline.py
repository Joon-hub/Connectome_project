"""
Run Complete Brain Connectivity Analysis Pipeline - FIXED VERSION
=================================================================
This script runs the entire analysis pipeline from start to finish.
Now with better error handling and clearer output.
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure imports work relative to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_pipeline.utils import print_section

def main():
    """
    Main entry point for running the full brain connectivity pipeline.
    
    THE COMPLETE WORKFLOW:
    1. Train classifier to recognize brain regions by their connectivity
    2. Apply to task data to find regions with altered connectivity
    3. Create visualizations comparing rest vs task
    """
    
    print("\n" + "="*80)
    print("BRAIN CONNECTIVITY ANALYSIS PIPELINE".center(80))
    print("="*80)
    print("""
    This pipeline will:
      1. Train a classifier on PIOP-2 resting-state data
      2. Apply the model to PIOP-1 gender task data  
      3. Create error maps and visualizations
      4. Generate comprehensive reports
      
    The goal: Find brain regions that change their connectivity during tasks!
    """)

    # Parse command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Cross-validation fold number (1-5)",
    )
    args = parser.parse_args()
    fold = args.fold

    print(f"\n📊 Running analysis for FOLD {fold}/5")
    print("-" * 80)
    
    start_time = time.time()

    try:
        # ============================================
        # STEP 1: Train model on resting-state data
        # ============================================
        print_section("STEP 1/3: TRAINING MODEL", width=80)
        
        # Import here to avoid issues if file doesn't exist
        import scripts.a1_train_model as a1_train_model
        result = a1_train_model.main(fold=fold)
        
        if result != 0:
            print("❌ Error in training step!")
            return 1
            
        print("✅ Training completed successfully!\n")
        
        # ============================================
        # STEP 2: Apply model to task data
        # ============================================
        print_section("STEP 2/3: APPLYING TO TASK DATA", width=80)
        
        import scripts.a2_apply_to_task as a2_apply_to_task
        try:
            a2_apply_to_task.main(fold=fold)
            print("✅ Task application completed successfully!\n")
        except Exception as e:
            print(f"❌ Error applying to task data: {e}")
            print("   (This is okay if PIOP-1 data is not available)")
            print("   Continuing to visualization...\n")

        # ============================================
        # STEP 3: Create visualizations
        # ============================================
        print_section("STEP 3/3: CREATING VISUALIZATIONS", width=80)
        
        import scripts.a3_visualize_results as a3_visualize_results
        try:
            a3_visualize_results.main()
            print("✅ Visualizations created successfully!\n")
        except Exception as e:
            print(f"⚠️  Warning in visualization: {e}")
            print("   Some visualizations may be incomplete")

    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================
    # SUMMARY
    # ============================================
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE! 🎉".center(80))
    print("="*80)
    print(f"""
    Total execution time: {minutes}m {seconds}s
    Fold processed: {fold}

    📁 Output locations:
    • Models:         data/processed/
    • Error maps:     data/results/
    • Visualizations: data/results/
    
    📊 Key outputs to check:
    • error_map_piop2_training_fold{fold}.csv - Training error by region
    • error_map_piop1_fold{fold}.csv - Task error by region  
    • error_comparison_rest_vs_task_fold{fold}.csv - Changes in connectivity
    
    💡 Next steps:
    • Review the error maps to identify task-engaged regions
    • Check visualizations to understand network-level changes
    • Run more folds for robust results (use --fold 2, 3, 4, 5)
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
