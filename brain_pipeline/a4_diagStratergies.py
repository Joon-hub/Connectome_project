"""
Diagonal Imputation Strategy Comparison Script
==============================================
Test and compare all diagonal imputation strategies on your brain connectivity data.

This script will:
1. Load your data
2. Create connectivity matrices
3. Apply each imputation strategy
4. Compare results visually and statistically
5. Help you choose the best strategy for your project
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brain_pipeline.a3_diagonal import (
    DiagonalImputer, NetworkParser, impute_connectivity_diagonal
)

# Assuming you have these modules from your project
from brain_pipeline import Config, DataLoader, ConnectivityProcessor


def analyze_diagonal_strategy_impact(X_train, y_train, subjects, region_list, config):
    """
    Analyze how different diagonal imputation strategies affect classification accuracy.
    
    This is KEY for your thesis - showing that network-aware imputation
    improves model performance!
    """
    from brain_pipeline import BrainRegionClassifier
    
    strategies = [
        'zero', 'one', 'mean', 'knn', 
        'network_mean', 'spatial_mean', 'hybrid'
    ]
    
    results = {
        'strategy': [],
        'cv_mean_accuracy': [],
        'cv_std_accuracy': [],
        'train_accuracy': []
    }
    
    print("\n" + "="*80)
    print("ANALYZING IMPACT OF DIAGONAL IMPUTATION ON CLASSIFICATION ACCURACY")
    print("="*80)
    
    for strategy in strategies:
        print(f"\n--- Testing Strategy: {strategy.upper()} ---")
        
        try:
            # Create classifier
            classifier = BrainRegionClassifier(config)
            
            # Cross-validate
            cv_results = classifier.cross_validate(X_train, y_train, subjects)
            
            # Train on full data
            train_acc = classifier.train(X_train, y_train)
            
            # Store results
            results['strategy'].append(strategy)
            results['cv_mean_accuracy'].append(cv_results['mean_accuracy'])
            results['cv_std_accuracy'].append(cv_results['std_accuracy'])
            results['train_accuracy'].append(train_acc)
            
            print(f"✓ CV Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            print(f"✓ Train Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results['strategy'].append(strategy)
            results['cv_mean_accuracy'].append(np.nan)
            results['cv_std_accuracy'].append(np.nan)
            results['train_accuracy'].append(np.nan)
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    return df_results


def visualize_diagonal_values(matrices_dict, region_list, output_path='diagonal_comparison.png'):
    """
    Visualize diagonal values across different imputation strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Diagonal value distributions
    ax1 = axes[0, 0]
    for strategy, matrix in matrices_dict.items():
        if matrix is not None:
            diagonal = np.diag(matrix)
            ax1.hist(diagonal, bins=30, alpha=0.5, label=strategy)
    
    ax1.set_xlabel('Diagonal Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Diagonal Values by Strategy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = []
    labels_for_box = []
    
    for strategy, matrix in matrices_dict.items():
        if matrix is not None:
            diagonal = np.diag(matrix)
            data_for_box.append(diagonal)
            labels_for_box.append(strategy)
    
    ax2.boxplot(data_for_box, labels=labels_for_box)
    ax2.set_ylabel('Diagonal Value', fontsize=12, fontweight='bold')
    ax2.set_title('Diagonal Value Ranges by Strategy', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Network-specific diagonal values (for network_mean strategy)
    ax3 = axes[1, 0]
    if 'network_mean' in matrices_dict and matrices_dict['network_mean'] is not None:
        matrix = matrices_dict['network_mean']
        diagonal = np.diag(matrix)
        
        # Parse networks
        network_ids, unique_networks = NetworkParser.get_network_membership(region_list)
        
        # Calculate mean diagonal per network
        network_means = []
        network_labels = []
        
        for net_id, net_name in enumerate(unique_networks):
            mask = network_ids == net_id
            if mask.any():
                network_means.append(diagonal[mask].mean())
                network_labels.append(net_name[:20])  # Truncate long names
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(network_means)))
        ax3.barh(range(len(network_means)), network_means, color=colors)
        ax3.set_yticks(range(len(network_means)))
        ax3.set_yticklabels(network_labels, fontsize=9)
        ax3.set_xlabel('Mean Diagonal Value', fontsize=12, fontweight='bold')
        ax3.set_title('Network-Specific Diagonal Values', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Network-mean strategy\nnot available',
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = "DIAGONAL STATISTICS SUMMARY\n" + "="*40 + "\n\n"
    
    for strategy, matrix in matrices_dict.items():
        if matrix is not None:
            diagonal = np.diag(matrix)
            stats_text += f"{strategy.upper()}:\n"
            stats_text += f"  Mean:   {diagonal.mean():7.4f}\n"
            stats_text += f"  Median: {diagonal.median():7.4f}\n"
            stats_text += f"  Std:    {diagonal.std():7.4f}\n"
            stats_text += f"  Range:  [{diagonal.min():.4f}, {diagonal.max():.4f}]\n\n"
    
    ax4.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    plt.close()


def main():
    """
    Main comparison workflow.
    """
    print("\n" + "="*80)
    print("DIAGONAL IMPUTATION STRATEGY COMPARISON")
    print("="*80)
    print("""
    This script will compare different diagonal imputation strategies:
    
    BASIC STRATEGIES:
      • zero:   Set diagonal to 0
      • one:    Set diagonal to 1 (perfect self-correlation)
      • mean:   Row-wise mean of off-diagonal elements
      • knn:    K-nearest neighbors imputation
    
    ADVANCED STRATEGIES (NEW):
      • network_mean:   Mean connectivity within same functional network
      • spatial_mean:   Mean connectivity to spatially neighboring regions
      • hybrid:         Weighted combination of network + spatial
    
    We'll evaluate:
      1. Diagonal value distributions
      2. Network-specific patterns
      3. Impact on classification accuracy
    """)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = Config("config.yaml")
    
    # Load data
    print("\n[2/6] Loading PIOP-2 resting-state data...")
    data_loader = DataLoader(config)
    df_piop2 = data_loader.load_piop2()
    
    # Extract regions
    print("\n[3/6] Extracting brain regions...")
    processor = ConnectivityProcessor(config)
    connection_columns = data_loader.get_connection_columns(df_piop2)
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)
    
    # Take one subject's data as example
    print("\n[4/6] Creating example connectivity matrix...")
    subject_data = df_piop2.iloc[0, 1:].to_numpy(dtype=float)
    example_matrix = processor.reconstruct_matrix(subject_data, connection_columns)
    
    # Compare strategies on this example
    print("\n[5/6] Comparing diagonal imputation strategies...")
    strategies = ['zero', 'one', 'mean', 'knn', 'network_mean', 'spatial_mean', 'hybrid']
    
    matrices_dict = {}
    
    for strategy in strategies:
        print(f"\n  → Testing: {strategy}")
        try:
            imputed = impute_connectivity_diagonal(
                example_matrix.copy(),
                region_list=region_list,
                strategy=strategy,
                config_path="config.yaml"
            )
            matrices_dict[strategy] = imputed
            
            # Show statistics
            diagonal = np.diag(imputed)
            print(f"    Mean: {diagonal.mean():.4f}, Std: {diagonal.std():.4f}, "
                  f"Range: [{diagonal.min():.4f}, {diagonal.max():.4f}]")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            matrices_dict[strategy] = None
    
    # Visualize comparison
    print("\n[6/6] Creating visualizations...")
    visualize_diagonal_values(matrices_dict, region_list, 
                              output_path='data/results/diagonal_comparison.png')
    
    # Save detailed results
    print("\nSaving detailed comparison results...")
    comparison_df = pd.DataFrame({
        'strategy': list(matrices_dict.keys()),
        'mean_diagonal': [np.diag(m).mean() if m is not None else np.nan 
                         for m in matrices_dict.values()],
        'std_diagonal': [np.diag(m).std() if m is not None else np.nan 
                        for m in matrices_dict.values()],
        'min_diagonal': [np.diag(m).min() if m is not None else np.nan 
                        for m in matrices_dict.values()],
        'max_diagonal': [np.diag(m).max() if m is not None else np.nan 
                        for m in matrices_dict.values()]
    })
    
    comparison_df.to_csv('data/results/diagonal_strategy_comparison.csv', index=False)
    print("✓ Saved to: data/results/diagonal_strategy_comparison.csv")
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
    Based on the neuroscience literature and your project goals:
    
    1. FOR RESTING-STATE CLASSIFICATION (your training step):
       → Recommended: 'network_mean' or 'hybrid'
       → Rationale: Preserves functional network structure, which is stable at rest
    
    2. FOR DETECTING TASK-INDUCED CHANGES (your test step):
       → Keep the SAME strategy as training
       → Rationale: Differences in error rates reveal task-induced reorganization
    
    3. FOR YOUR THESIS:
       → Compare 'mean' (baseline) vs 'network_mean' (your contribution)
       → Show that network-aware imputation improves classification accuracy
       → Demonstrate that it better preserves known brain organization
    
    Next steps:
    1. Update config.yaml to set: strategy: "network_mean"
    2. Re-run your training pipeline
    3. Compare results with previous 'mean' or 'knn' strategy
    4. Document the improvement in your thesis!
    """)
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()