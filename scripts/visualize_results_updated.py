'''
Create Comprehensive Visualizations
==================================
Updated version: Works with final aggregated results instead of per-fold outputs
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from brain_pipeline import Config, Visualizer
from brain_pipeline.utils import print_section


def main():
    print_section("CREATING VISUALIZATIONS")
    
    # Load configuration
    config = Config()
    visualizer = Visualizer(config)
    
    # === [1/4] Load results ===
    print("\n[1/4] Loading results...")
    
    results_dir = Path("data/results")
    
    # Check for final result files
    train_file = results_dir / "error_map_training_final.csv"
    test_file = results_dir / "error_map_task_final.csv"
    comparison_file = results_dir / "error_comparison_rest_vs_task_final.csv"
    
    if not test_file.exists():
        print(f"ERROR: Task results not found at {test_file}")
        print("Please run apply_to_task.py first!")
        return 1
    
    # Load test results (required)
    error_map_test = pd.read_csv(test_file)
    print(f"Loaded task results: {len(error_map_test)} regions")
    
    # Load training results (optional)
    if train_file.exists():
        error_map_train = pd.read_csv(train_file)
        print(f"Loaded training results: {len(error_map_train)} regions")
    else:
        error_map_train = None
        print("Training results not found - skipping rest vs task comparison")
    
    # Load comparison (optional)
    if comparison_file.exists():
        comparison = pd.read_csv(comparison_file)
        print(f"Loaded comparison: {len(comparison)} regions")
    else:
        comparison = None
        print("Comparison file not found - will skip comparison plots")
    
    # === [2/4] Create error map visualizations ===
    print("\n[2/4] Creating error map visualizations...")
    
    # Training error map (if available)
    if error_map_train is not None:
        fig1 = visualizer.plot_error_map(error_map_train, "PIOP-2 Training Error Map")
        visualizer.save_figure(fig1, 'error_map_piop2_final.png')
    
    # Test error map (always available)
    fig2 = visualizer.plot_error_map(error_map_test, "PIOP-1 Task Error Map")
    visualizer.save_figure(fig2, 'error_map_piop1_final.png')
    
    # === [3/4] Comparison visualization ===
    if comparison is not None and error_map_train is not None:
        print("\n[3/4] Creating comparison visualizations...")
        fig3 = create_comparison_plot(error_map_train, error_map_test, comparison, config)
        visualizer.save_figure(fig3, 'comparison_rest_vs_task_final.png')
    else:
        print("\n[3/4] Skipping comparison visualization (missing data)")
    
    # === [4/4] Network-level analysis ===
    print("\n[4/4] Creating network-level analysis...")
    fig4 = create_network_analysis(error_map_test, config)
    visualizer.save_figure(fig4, 'network_analysis_final.png')
    
    # === Summary ===
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    if error_map_train is not None:
        print(" ✓ error_map_piop2_final.png")
    print(" ✓ error_map_piop1_final.png")
    if comparison is not None:
        print(" ✓ comparison_rest_vs_task_final.png")
    print(" ✓ network_analysis_final.png")
    
    return 0


def create_comparison_plot(error_train, error_test, comparison, config):
    '''Create comparison visualization between rest and task'''
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ✅ Align datasets by region name
    merged = pd.merge(
        error_train[['region_name', 'misclassification_rate']],
        error_test[['region_name', 'misclassification_rate']],
        on='region_name',
        suffixes=('_train', '_test')
    )

    # Plot 1: Scatter comparison
    ax1 = axes[0, 0]
    ax1.scatter(
        merged['misclassification_rate_train'],
        merged['misclassification_rate_test'],
        alpha=0.5, s=50
    )
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Error')
    ax1.set_xlabel('Resting-State Error', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Task Error', fontsize=12, fontweight='bold')
    ax1.set_title('Error Rate Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Top altered regions
    ax2 = axes[0, 1]
    top_20_altered = comparison.head(20)
    colors = ['red' if x > 0.1 else 'orange' for x in top_20_altered['error_increase']]
    ax2.barh(range(20), top_20_altered['error_increase'], color=colors,
             edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(
        [name[:35] for name in top_20_altered['region_name']],
        fontsize=9
    )
    ax2.set_xlabel('Error Increase (Task - Rest)', fontsize=12, fontweight='bold')
    ax2.set_title('Top 20 Altered Regions', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Error increase distribution
    ax3 = axes[1, 0]
    ax3.hist(comparison['error_increase'], bins=50, alpha=0.7,
             color='steelblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3.axvline(
        comparison['error_increase'].mean(), color='green',
        linestyle='--', linewidth=2, label=f"Mean: {comparison['error_increase'].mean():.3f}"
    )
    ax3.set_xlabel('Error Increase', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Regions', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Error Changes', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    n_increased = (comparison['error_increase'] > 0.05).sum()
    n_decreased = (comparison['error_increase'] < -0.05).sum()
    n_stable = len(comparison) - n_increased - n_decreased
    
    stats_text = f"""
    COMPARISON SUMMARY
    ==================
    
    Total Regions: {len(comparison)}
    
    Error Changes:
      ↑ Increased (>5%):  {n_increased}
      ↓ Decreased (>5%):  {n_decreased}
      ≈ Stable:           {n_stable}
    
    Mean Error Increase: {comparison['error_increase'].mean():.4f}
    Max Error Increase:  {comparison['error_increase'].max():.4f}
    Min Error Increase:  {comparison['error_increase'].min():.4f}
    
    Overall Performance:
      Rest Mean Error: {error_train['misclassification_rate'].mean():.4f}
      Task Mean Error: {error_test['misclassification_rate'].mean():.4f}
      Difference:      {error_test['misclassification_rate'].mean() - error_train['misclassification_rate'].mean():.4f}
    """
    
    ax4.text(
        0.1, 0.5, stats_text, fontsize=11, family='monospace',
        verticalalignment='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    plt.tight_layout()
    return fig


def create_network_analysis(error_map, config):
    """Create cortical–subcortical network-level analysis visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Parse atlas and networks
    def parse_atlas_and_networks(region_name):
        if region_name.startswith(('LH_', 'RH_')):
            atlas = 'Schaefer'
            schaefer_map = {
                'VisCent': ('Visual', 'Visual Central'),
                'VisPeri': ('Visual', 'Visual Peripheral'),
                'SomMotA': ('Somatomotor', 'Somatomotor A'),
                'SomMotB': ('Somatomotor', 'Somatomotor B'),
                'DorsAttnA': ('Dorsal Attention', 'DorsAttn A'),
                'DorsAttnB': ('Dorsal Attention', 'DorsAttn B'),
                'SalVentAttnA': ('Salience/Ventral Attention', 'SalVentAttn A'),
                'SalVentAttnB': ('Salience/Ventral Attention', 'SalVentAttn B'),
                'LimbicA': ('Limbic', 'Limbic A'),
                'LimbicB': ('Limbic', 'Limbic B'),
                'ContA': ('Control', 'Control A'),
                'ContB': ('Control', 'Control B'),
                'ContC': ('Control', 'Control C'),
                'DefaultA': ('Default Mode', 'Default A'),
                'DefaultB': ('Default Mode', 'Default B'),
                'DefaultC': ('Default Mode', 'Default C'),
                'TempPar': ('Temporal-Parietal', 'TempPar')
            }
            for key, (net7, net17) in schaefer_map.items():
                if key in region_name:
                    return atlas, net7, net17
            return atlas, 'Other', 'Other'
        else:
            atlas = 'Tian'
            subcortex_map = {
                'THA': 'Thalamus',
                'CAU': 'Caudate',
                'pCAU': 'Caudate Posterior',
                'aCAU': 'Caudate Anterior',
                'PUT': 'Putamen',
                'pPUT': 'Putamen Posterior',
                'aPUT': 'Putamen Anterior',
                'GP': 'Globus Pallidus',
                'pGP': 'Posterior GP',
                'aGP': 'Anterior GP',
                'HIP': 'Hippocampus',
                'pHIP': 'Posterior Hippocampus',
                'aHIP': 'Anterior Hippocampus',
                'AMY': 'Amygdala',
                'lAMY': 'Lateral Amygdala',
                'mAMY': 'Medial Amygdala',
                'NAc': 'Nucleus Accumbens'
            }
            for key, val in subcortex_map.items():
                if key in region_name:
                    return atlas, 'Subcortex', val
            return atlas, 'Subcortex', 'Other'

    error_map[['atlas', 'network_7', 'network_17']] = error_map['region_name'].apply(
        lambda r: pd.Series(parse_atlas_and_networks(r))
    )

    error_map['hemisphere'] = error_map['region_name'].apply(
        lambda x: 'Left' if x.startswith('LH_') or x.endswith('-lh')
        else 'Right' if x.startswith('RH_') or x.endswith('-rh')
        else 'Unknown'
    )

    network_version = config.get("network_version", "7")
    network_column = "network_7" if network_version == "7" else "network_17"
    title_suffix = " (7-Network Schaefer)" if network_version == "7" else " (17-Network Schaefer)"

    # Plot 1: Network-level errors
    ax1 = axes[0, 0]
    network_errors = (
        error_map.groupby(network_column)['misclassification_rate']
        .mean().sort_values(ascending=True)
    )
    colors = plt.cm.RdYlGn_r(network_errors.values / network_errors.max())
    ax1.barh(range(len(network_errors)), network_errors.values,
             color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(network_errors)))
    ax1.set_yticklabels(network_errors.index, fontsize=10)
    ax1.set_xlabel('Mean Misclassification Rate', fontsize=11, fontweight='bold')
    ax1.set_title(f'Error Rate by Network{title_suffix}', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Hemisphere comparison
    ax2 = axes[0, 1]
    hemisphere_errors = error_map.groupby('hemisphere')['misclassification_rate'].mean()
    bars = ax2.bar(
        hemisphere_errors.index, hemisphere_errors.values,
        color=['#FF6B6B', '#4ECDC4', '#95E1D3'],
        edgecolor='black', linewidth=2, alpha=0.7
    )
    ax2.set_ylabel('Mean Misclassification Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Error Rate by Hemisphere', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Atlas comparison
    ax3 = axes[1, 0]
    sns.violinplot(
        data=error_map, y='atlas', x='misclassification_rate', hue='atlas',
        ax=ax3, palette='coolwarm', legend=False
    )
    ax3.set_xlabel('Misclassification Rate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Atlas (Cortex vs Subcortex)', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution by Atlas', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Plot 4: Atlas-level summary
    ax4 = axes[1, 1]
    summary = (
        error_map.groupby('atlas')['misclassification_rate']
        .agg(['mean', 'std', 'count']).reset_index()
    )
    ax4.axis('off')
    stats_text = "ATLAS-LEVEL SUMMARY\n===================\n"
    for _, row in summary.iterrows():
        stats_text += f"{row['atlas']}: mean={row['mean']:.3f}, std={row['std']:.3f}, n={int(row['count'])}\n"
    ax4.text(
        0.05, 0.5, stats_text,
        fontsize=13,
        family='monospace',
        verticalalignment='center',
        fontweight='bold',
        linespacing=1.5,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    sys.exit(main())