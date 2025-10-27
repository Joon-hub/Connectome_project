"""
Train Brain Region Classifier on PIOP-2 Resting-State fMRI Data
================================================================
"""

import sys 
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from brain_pipeline import (
    Config, DataLoader, ConnectivityProcessor, 
    BrainRegionClassifier, ModelEvaluator
)
from brain_pipeline.utils import print_section

def main(fold: int = 1):
    """
    Train a brain region classifier on PIOP-2 resting-state fMRI data.
    
    THE BIG IDEA:
    - We have 232 brain regions
    - Each region connects to all other regions
    - We train a model to recognize regions based on WHO they connect to
    - It's like identifying someone based on their friend network
    """
    
    print_section(f"STEP 1: Training Model on PIOP-2 (Resting-State) | Fold {fold}")
    
    # Load configuration
    config = Config()
    
    # Initialize components
    data_loader = DataLoader(config)
    processor = ConnectivityProcessor(config)
    
    # ============================================
    # STEP 1: Load the data
    # ============================================
    print("\n[1/5] Loading PIOP-2 resting-state fMRI data...")
    df_piop2 = data_loader.load_piop2()
    connection_columns = data_loader.get_connection_columns(df_piop2)
    print(f"  → Loaded {len(df_piop2)} subjects")
    print(f"  → Found {len(connection_columns)} connections")
    
    # ============================================
    # STEP 2: Extract brain regions   
    # ============================================
    print("\n[2/5] Extracting brain regions from connection columns...")
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)
    print(f"  → Found {n_regions} unique brain regions")
    
    # Save region list (only on first fold to avoid overwriting)
    if fold == 1:
        # Save as simple text file for easy inspection
        region_file = Path("data/processed/brain_regions.txt")
        with open(region_file, 'w') as f:
            for i, region in enumerate(region_list):
                f.write(f"{i}: {region}\n")
        print(f"  → Saved region list to {region_file}")
    
    # ============================================
    # STEP 3: Create training dataset
    # ============================================
    print("\n[3/5] Creating dataset for model training...")
    X_train, y_train, subjects = processor.create_dataset(df_piop2, connection_columns)
    
    print(f"  → Dataset shape: {X_train.shape}")
    print(f"  → Number of samples: {len(X_train)} ({len(df_piop2)} subjects × {n_regions} regions)")
    print(f"  → Features per sample: {X_train.shape[1]} (connections to {n_regions-1} other regions)")
    print(f"  → Labels: {len(np.unique(y_train))} unique region IDs")
    
    # Save processed data (only on first fold)
    if fold == 1:
        print("\n  Saving processed data for inspection...")
        
        # Create meaningful column names for the features
        # Each column represents a connection to another region
        feature_columns = []
        for target_region in region_list:
            # Skip the region itself (diagonal was removed)
            other_regions = [r for r in region_list if r != target_region]
            # For simplicity in this dataset, we can't directly map which column
            # corresponds to which region due to the diagonal removal
            # So we'll just use generic names
            break
        
        # Simple column names: conn_0, conn_1, etc.
        feature_columns = [f"conn_{i}" for i in range(X_train.shape[1])]
        
        # Create DataFrame with data
        df_processed = pd.DataFrame(X_train, columns=feature_columns)
        df_processed["region_label"] = y_train
        df_processed["subject_id"] = subjects
        
        # Add region name for readability
        df_processed["region_name"] = [region_list[label] for label in y_train]
        
        # Save to CSV
        data_file = Path("data/processed/training_data_clean.csv")
        df_processed.to_csv(data_file, index=False)
        print(f"  → Saved processed data to {data_file}")
        print(f"    (First few columns: {feature_columns[:3]} ... {feature_columns[-1]})")
    
    # ============================================
    # STEP 4: Train model with cross-validation
    # ============================================
    print("\n[4/5] Training model with cross-validation...")
    classifier = BrainRegionClassifier(config)
    
    # Run cross-validation to see how well model generalizes
    cv_results = classifier.cross_validate(X_train, y_train, subjects)
    
    print(f"\n  Cross-validation results:")
    print(f"  → Mean accuracy: {cv_results['mean_accuracy']:.4f}")
    print(f"  → Std deviation: {cv_results['std_accuracy']:.4f}")
    print(f"  → All fold accuracies: {[f'{acc:.4f}' for acc in cv_results['fold_accuracies']]}")
    
    # Train final model on full dataset
    print("\n  Training final model on full dataset...")
    train_accuracy = classifier.train(X_train, y_train)
    
    # ============================================
    # STEP 5: Evaluate and save
    # ============================================
    print("\n[5/5] Evaluating model and saving results...")
    
    # Get predictions on training data
    y_train_pred, _ = classifier.predict(X_train)
    
    # Calculate error map (which regions are hardest to classify)
    evaluator = ModelEvaluator(config, region_list)
    error_map_train = evaluator.calculate_error_map(y_train, y_train_pred)
    
    # Save model
    model_path = f"data/processed/trained_model_fold{fold}.pkl"
    classifier.save(model_path)
    
    # Save error map
    error_file = f"error_map_piop2_training_fold{fold}.csv"
    evaluator.save_results(error_map_train, error_file)
    
    # ============================================
    # SUMMARY
    # ============================================
    stats = evaluator.get_summary_stats(error_map_train)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"Fold: {fold}")
    print(f"Number of regions: {n_regions}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Mean error rate by region: {stats['mean_error']:.4f}")
    print(f"Regions with high error (>{stats.get('high_threshold', 0.3):.0%}): {stats['n_high_error']}")
    print(f"Regions with low error (<{stats.get('low_threshold', 0.1):.0%}): {stats['n_low_error']}")
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: data/results/{error_file}")
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Brain Region Classifier on PIOP-2 Resting-State fMRI Data"
    )
    parser.add_argument(
        '--fold', 
        type=int, 
        default=1, 
        help='Cross-validation fold number (1-5)'
    )
    args = parser.parse_args()
    
    sys.exit(main(fold=args.fold))