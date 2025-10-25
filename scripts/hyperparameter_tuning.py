"""
Hyperparameter Tuning for Brain Region Classifier
==================================================
This script performs grid search for optimal logistic regression parameters.
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

# Ensure imports work relative to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_pipeline import (
    Config, DataLoader, ConnectivityProcessor,
    BrainRegionClassifier, ModelEvaluator
)
from brain_pipeline.utils import print_section


def main(job_id: int = 0):
    """
    Perform hyperparameter tuning with a specific parameter combination.
    
    Parameters
    ----------
    job_id : int
        Job index to select which hyperparameter combination to test
    """
    print_section(f"HYPERPARAMETER TUNING - JOB {job_id}")
    
    # Load configuration
    config = Config()
    
    # Define hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [500, 1000, 2000]
    }
    
    # Generate all combinations
    param_combinations = list(product(
        param_grid['C'],
        param_grid['penalty'],
        param_grid['solver'],
        param_grid['max_iter']
    ))
    
    print(f"\nTotal parameter combinations: {len(param_combinations)}")
    
    # Select parameters for this job
    if job_id >= len(param_combinations):
        print(f"ERROR: Job ID {job_id} exceeds number of combinations ({len(param_combinations)})")
        return 1
    
    C, penalty, solver, max_iter = param_combinations[job_id]
    
    print(f"\nTesting parameters:")
    print(f"  C: {C}")
    print(f"  Penalty: {penalty}")
    print(f"  Solver: {solver}")
    print(f"  Max iterations: {max_iter}")
    
    # Initialize components
    data_loader = DataLoader(config)
    processor = ConnectivityProcessor()
    
    # Load and process data
    print("\n[1/4] Loading and processing data...")
    df_piop2 = data_loader.load_piop2()
    connection_columns = data_loader.get_connection_columns(df_piop2)
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)
    X, y, subjects = processor.create_dataset(df_piop2, connection_columns)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Update model parameters
    print("\n[2/4] Initializing model with test parameters...")
    config.config['model']['params']['C'] = C
    config.config['model']['params']['penalty'] = penalty
    config.config['model']['params']['solver'] = solver
    config.config['model']['params']['max_iter'] = max_iter
    
    # Create classifier
    classifier = BrainRegionClassifier(config)
    
    # Perform cross-validation
    print("\n[3/4] Performing 5-fold cross-validation...")
    start_time = time.time()
    cv_results = classifier.cross_validate(X, y, subjects)
    cv_time = time.time() - start_time
    
    # Extract results
    mean_accuracy = cv_results['mean_accuracy']
    std_accuracy = cv_results['std_accuracy']
    fold_accuracies = cv_results['fold_accuracies']
    
    print(f"\nCross-validation complete!")
    print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"  CV time: {cv_time:.2f}s")
    
    # Save results
    print("\n[4/4] Saving results...")
    results = {
        'job_id': job_id,
        'parameters': {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter
        },
        'cv_results': {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'fold_accuracies': [float(acc) for acc in fold_accuracies],
            'cv_time_seconds': float(cv_time)
        },
        'dataset_info': {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_regions': int(n_regions),
            'n_subjects': int(len(np.unique(subjects)))
        }
    }
    
    # Save to results directory
    results_dir = Path(config.get('data', 'results_dir')) / 'hyperparameter_tuning'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f'tuning_job_{job_id:03d}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Also save as CSV for easier aggregation
    csv_file = results_dir / f'tuning_job_{job_id:03d}.csv'
    result_df = pd.DataFrame([{
        'job_id': job_id,
        'C': C,
        'penalty': penalty,
        'solver': solver,
        'max_iter': max_iter,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'cv_time': cv_time
    }])
    result_df.to_csv(csv_file, index=False)
    
    print(f"CSV saved to: {csv_file}")
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING JOB COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for brain region classifier'
    )
    parser.add_argument(
        '--job-id',
        type=int,
        default=0,
        help='Job ID for parameter combination selection'
    )
    args = parser.parse_args()
    
    sys.exit(main(job_id=args.job_id))