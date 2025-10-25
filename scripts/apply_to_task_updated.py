"""
Apply Trained Model to PIOP-1 Task Data
=======================================
Updated version: Uses final trained model without per-fold outputs
"""

import sys
from pathlib import Path

# Ensure imports work relative to project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_pipeline import (
    Config, DataLoader, ConnectivityProcessor,
    BrainRegionClassifier, ModelEvaluator
)
from brain_pipeline.utils import print_section
import pandas as pd


def main():
    """Apply a trained model to PIOP-1 gender task data."""
    print_section("APPLYING MODEL TO PIOP-1 (GENDER TASK)")

    # Load configuration and components
    config = Config()
    data_loader = DataLoader(config)
    processor = ConnectivityProcessor()

    # Load PIOP-2 to get region info
    print("\n[1/6] Loading region information...")
    df_piop2 = data_loader.load_piop2()
    connection_columns = data_loader.get_connection_columns(df_piop2)
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)

    # Load model
    print("\n[2/6] Loading trained model...")
    classifier = BrainRegionClassifier(config)
    model_path = "data/processed/trained_model_final.pkl"
    
    if not Path(model_path).exists():
        print(f"ERROR: Trained model not found at {model_path}")
        print("Please run train_model.py first!")
        return 1
    
    classifier.load(model_path)

    # Load PIOP-1 data
    print("\n[3/6] Loading PIOP-1 task data...")
    df_piop1 = data_loader.load_piop1()

    # Verify columns match
    piop1_columns = data_loader.get_connection_columns(df_piop1)
    assert piop1_columns == connection_columns, "Column mismatch between datasets!"
    print("Column structure verified")

    # Create test dataset
    print("\n[4/6] Creating test dataset...")
    X_test, y_test, subjects_test = processor.create_dataset(df_piop1, connection_columns)

    # Apply model
    print("\n[5/6] Applying model to task data...")
    y_test_pred, y_test_proba = classifier.predict(X_test)

    # Create error map
    print("\n[6/6] Creating error map...")
    evaluator = ModelEvaluator(config, region_list)
    error_map_test = evaluator.calculate_error_map(y_test, y_test_pred)

    # Save test results
    evaluator.save_results(error_map_test, "error_map_task_final.csv")

    # Load training error map for comparison (if exists)
    train_error_file = "data/results/error_map_training_final.csv"
    if Path(train_error_file).exists():
        error_map_train = pd.read_csv(train_error_file)
        
        # Compare datasets
        comparison = evaluator.compare_datasets(error_map_train, error_map_test)
        comparison.to_csv("data/results/error_comparison_rest_vs_task_final.csv", index=False)
        print(f"Comparison saved to: data/results/error_comparison_rest_vs_task_final.csv")
        
        # Summary comparison
        stats_test = evaluator.get_summary_stats(error_map_test)
        stats_train = evaluator.get_summary_stats(error_map_train)

        print("\n" + "=" * 70)
        print("APPLICATION COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"\nResting-State (PIOP-2):")
        print(f"  Mean error rate: {stats_train['mean_error']:.4f}")
        print(f"  High error regions: {stats_train['n_high_error']}")
        print(f"\nGender Task (PIOP-1):")
        print(f"  Mean error rate: {stats_test['mean_error']:.4f}")
        print(f"  High error regions: {stats_test['n_high_error']}")
        print(f"\nError Increase:")
        print(f"  Mean error: {stats_test['mean_error'] - stats_train['mean_error']:.4f}")
        print(f"  High error regions: {stats_test['n_high_error'] - stats_train['n_high_error']}")
    else:
        print("\nTraining error map not found. Skipping comparison.")
        stats_test = evaluator.get_summary_stats(error_map_test)
        print("\n" + "=" * 70)
        print("APPLICATION COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"\nGender Task (PIOP-1):")
        print(f"  Mean error rate: {stats_test['mean_error']:.4f}")
        print(f"  High error regions: {stats_test['n_high_error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())