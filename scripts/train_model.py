"""
Train Brain Region Classifier on PIOP-2- Resting-State fMRI Data
================================================================
"""

import re
import sys 
from pathlib import Path
import pandas as pd

from brain_pipeline.model import BrainRegionClassifier

# Ensure import works relative to project root
sys.path.append(str(Path(__file__).parent.parent))

from brain_pipeline import (Config, DataLoader, ConnectivityProcessor, BrainRegionClassifier, ModelEvaluator)
from brain_pipeline.utils import print_section

def main(fold: int = 1):
    """ Train a brain region classifier on PIOP-2 resting-state fMRI data. """
    
    print_section(f"STEP 1: Training Model on PIOP-2 (Resting-State) | Fold {fold}")
    # Load configuration
    config = Config()
    
    # Intialize components
    data_loader = DataLoader(config)
    processor = ConnectivityProcessor()
    
    # 1. Load data
    print(" \n [1/5] Loading PIOP-2 resting-state fMRI data...")
    df_piop2 = data_loader.load_piop2()
    connection_columns = data_loader.get_connection_columns(df_piop2)
    
    # Save connection metadata (only on first fold)
    conn_file = Path("data/processed/region_connections.csv")
    if fold == 1:
        processor.save_connection_columns(connection_columns, conn_file)
    
    
    
    # 2. Extract brain regions   
    print(" \n [2/5] Extracting brain regions from connection data...")
    region_list, _, _ = processor.extract_regions(connection_columns)
    
    # save region_list (only on first fold) as csv 
    region_list = pd.DataFrame(region_list, columns=["Region"])
    region_file = Path("data/processed/brain_regions.csv")
    if fold == 1:
        region_list.to_csv(region_file, index=False)
        print(f" Saved extracted brain regions to {region_file}")
    
    
    # 3. Create dataset
    print(" \n [3/5] Creating dataset for model training...")
    X_train, y_train , subjects = processor.create_dataset(df_piop2, connection_columns)
    
    # 4. Train model with cross-validation
    print(" \n [4/5] Training model with cross-validation...")
    classifier = BrainRegionClassifier(config)
    
    cv_results = classifier.cross_validate(X_train, y_train, subjects)
    print(f" Completed training for fold {fold}. mean accuracy: {cv_results['mean_accuracy']:.4f} +/- {cv_results['std_accuracy']:.4f} ")

    # Train final model on full dataset
    train_accuracy = classifier.train(X_train, y_train)
    
    # 5. Evaluate on training data
    print(" \n [5/5] Evaluating model on training data...")
    y_train_pred,_ = classifier.predict(X_train)
    evaluator = ModelEvaluator(config)
    error_map_train = evaluator.calculate_error_map(y_train, y_train_pred)
    
    # save results
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"brain_region_classifier_fold{fold}.pkl"
    
    # save model and results
    classifier.save(model_path)
    evaluator.save_results(error_map_train, model_dir / f"error_map_train_fold{fold}.csv")
    
    
    # summary 
    stats = evaluator.get_summary_stats(error_map_train)
    print("\n" + "="*70)
    print("Training complete summary:")
    print(f" Training Accuracy: {train_accuracy:.4f}")
    print(f" Mean Region-wise Error Rate: {stats['mean_error']:.4f}")
    print(f"Mean Error rate: {stats['mean_error']:.4f}")
    print(f" High error regions: {stats['n_high_error']}")
    print(f" Low error regions: {stats['n_low_error']}")
    print(f"\n Model saved to: {model_path.resolve()}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Brain Region Classifier on PIOP-2 Resting-State fMRI Data")
    parser.add_argument('--fold', type=int, default=1, help='Cross-validation fold number (default: 1)')
    args = parser.parse_args()
    
    main(fold=args.fold)