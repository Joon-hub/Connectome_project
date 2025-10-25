"""
Analyze Hyperparameter Tuning Results
=====================================
This script aggregates results from all tuning jobs and identifies best parameters.
"""

import sys
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure imports work relative to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_pipeline import Config
from brain_pipeline.utils import print_section


def main():
    """Aggregate and analyze hyperparameter tuning results."""
    print_section("HYPERPARAMETER TUNING ANALYSIS")
    
    # Load configuration
    config = Config()
    results_dir = Path(config.get('data', 'results_dir')) / 'hyperparameter_tuning'
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1
    
    # Find all CSV result files
    csv_files = sorted(glob.glob(str(results_dir / 'tuning_job_*.csv')))
    
    if not csv_files:
        print(f"ERROR: No tuning result files found in {results_dir}")
        return 1
    
    print(f"\nFound {len(csv_files)} tuning result files")
    
    # Load and concatenate all results
    print("\n[1/4] Loading results...")
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)
    
    results_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(results_df)} hyperparameter combinations")
    
    # Sort by mean accuracy
    results_df = results_df.sort_values('mean_accuracy', ascending=False)
    
    # Display top 10 configurations
    print("\n[2/4] Top 10 Configurations:")
    print("="*90)
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  Job ID: {row['job_id']}")
        print(f"  Parameters: C={row['C']}, solver={row['solver']}, max_iter={row['max_iter']}")
        print(f"  Mean Accuracy: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")
        print(f"  CV Time: {row['cv_time']:.2f}s")
    
    # Find best parameters
    best_row = results_df.iloc[0]
    print("\n" + "="*90)
    print("BEST CONFIGURATION:")
    print("="*90)
    print(f"Job ID: {best_row['job_id']}")
    print(f"Parameters:")
    print(f"  C: {best_row['C']}")
    print(f"  Penalty: {best_row['penalty']}")
    print(f"  Solver: {best_row['solver']}")
    print(f"  Max iterations: {best_row['max_iter']}")
    print(f"\nPerformance:")
    print(f"  Mean Accuracy: {best_row['mean_accuracy']:.4f} ± {best_row['std_accuracy']:.4f}")
    print(f"  CV Time: {best_row['cv_time']:.2f}s")
    
    # Save aggregated results
    print("\n[3/4] Saving aggregated results...")
    output_file = results_dir / 'hyperparameter_tuning_summary.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}")
    
    # Save best parameters as JSON
    best_params = {
        'best_job_id': int(best_row['job_id']),
        'best_parameters': {
            'C': float(best_row['C']),
            'penalty': str(best_row['penalty']),
            'solver': str(best_row['solver']),
            'max_iter': int(best_row['max_iter'])
        },
        'best_performance': {
            'mean_accuracy': float(best_row['mean_accuracy']),
            'std_accuracy': float(best_row['std_accuracy']),
            'cv_time': float(best_row['cv_time'])
        }
    }
    
    best_params_file = results_dir / 'best_parameters.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best parameters saved to: {best_params_file}")
    
    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    create_visualizations(results_df, results_dir)
    
    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)
    
    return 0


def create_visualizations(results_df, results_dir):
    """Create comprehensive visualizations of tuning results."""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy by C parameter
    ax1 = axes[0, 0]
    c_grouped = results_df.groupby('C')['mean_accuracy'].agg(['mean', 'std'])
    c_grouped.plot(y='mean', yerr='std', kind='bar', ax=ax1, 
                   color='steelblue', capsize=4, alpha=0.7)
    ax1.set_xlabel('Regularization Parameter (C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Regularization (C)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(['Mean ± Std'], loc='lower right')
    
    # Plot 2: Accuracy by solver
    ax2 = axes[0, 1]
    solver_grouped = results_df.groupby('solver')['mean_accuracy'].agg(['mean', 'std'])
    solver_grouped.plot(y='mean', yerr='std', kind='bar', ax=ax2,
                        color='coral', capsize=4, alpha=0.7)
    ax2.set_xlabel('Solver', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Solver', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(['Mean ± Std'], loc='lower right')
    
    # Plot 3: Accuracy vs computation time
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['cv_time'], results_df['mean_accuracy'],
                          c=results_df['C'], cmap='viridis', s=100, alpha=0.6,
                          edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('CV Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy vs Computation Time', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('C parameter', fontsize=10, fontweight='bold')
    
    # Plot 4: Heatmap of accuracy by C and solver
    ax4 = axes[1, 1]
    pivot_table = results_df.pivot_table(
        values='mean_accuracy',
        index='solver',
        columns='C',
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdYlGn',
                ax=ax4, cbar_kws={'label': 'Mean Accuracy'})
    ax4.set_xlabel('Regularization (C)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Solver', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy Heatmap: Solver vs C', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = results_dir / 'hyperparameter_tuning_analysis.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close(fig)
    
    # Create additional plot: Top 20 configurations
    fig2, ax = plt.subplots(figsize=(14, 8))
    top_20 = results_df.head(20)
    
    # Create labels combining parameters
    labels = [
        f"C={row['C']}\n{row['solver']}\niter={row['max_iter']}"
        for _, row in top_20.iterrows()
    ]
    
    colors = plt.cm.RdYlGn(top_20['mean_accuracy'] / top_20['mean_accuracy'].max())
    bars = ax.barh(range(20), top_20['mean_accuracy'], color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_yticks(range(20))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Hyperparameter Configurations', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc, std) in enumerate(zip(bars, top_20['mean_accuracy'], top_20['std_accuracy'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}±{std:.4f}',
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_file2 = results_dir / 'top_20_configurations.png'
    fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Top 20 plot saved to: {output_file2}")
    plt.close(fig2)


if __name__ == "__main__":
    sys.exit(main())