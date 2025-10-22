"""
Train Brain Region Classifier on PIOP-2 Resting-State Data
==========================================================
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


def main(fold: int = 1):
    """Train a brain region classifier on PIOP-2 resting-state data."""
    print_section(f"STEP 1: TRAINING MODEL ON PIOP-2 (RESTING-STATE) | FOLD {fold}")

    # Load configuration
    config = Config()

    # Initialize components
    data_loader = DataLoader(config)
    processor = ConnectivityProcessor()

    # 1. Load data
    print("\n[1/5] Loading PIOP-2 data...")
    df_piop2 = data_loader.load_piop2()
    connection_columns = data_loader.get_connection_columns(df_piop2)

    # Save connection metadata (only once)
    conn_file = Path("data/processed/region_connections.csv")
    if fold == 1 and not conn_file.exists():
        processor.save_connection_columns(connection_columns, conn_file)
    else:
        print(f"Using existing connection metadata: {conn_file}")

    # 2. Extract brain regions
    print("\n[2/5] Extracting brain regions...")
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)

    # Save region list (only once)
    region_file = Path("data/processed/region_list.csv")
    if fold == 1 and not region_file.exists():
        processor.save_region_list(region_list, region_file)
    else:
        print(f"Using existing region list: {region_file}")

    # 3. Create dataset
    print("\n[3/5] Creating training dataset...")
    X_train, y_train, subjects = processor.create_dataset(df_piop2, connection_columns,)

    # 4. Train model with cross-validation
    print("\n[4/5] Training classifier...")
    classifier = BrainRegionClassifier(config)

    cv_results = classifier.cross_validate(X_train, y_train, subjects)
    print("\nCross-validation results:")
    print(f"  Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")

    # Train final model on full data
    train_accuracy = classifier.train(X_train, y_train)

    # 5. Evaluate on training data
    print("\n[5/5] Creating training error map...")
    y_train_pred, _ = classifier.predict(X_train)
    evaluator = ModelEvaluator(config, region_list)
    error_map_train = evaluator.calculate_error_map(y_train, y_train_pred)

    # Save results
    model_path = Path(f"data/processed/trained_model_fold{fold}.pkl")
    classifier.save(model_path)

    evaluator.save_results(error_map_train, f"error_map_piop2_training_fold{fold}.csv")

    # Summary
    stats = evaluator.get_summary_stats(error_map_train)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Mean error rate: {stats['mean_error']:.4f}")
    print(f"High error regions: {stats['n_high_error']}")
    print(f"Low error regions: {stats['n_low_error']}")
    print(f"\nModel saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
