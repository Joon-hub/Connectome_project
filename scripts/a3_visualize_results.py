'''
Create Comprehensive Visualizations
==================================
'''

import sys
from pathlib import Path
import glob
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from brain_pipeline import Config, Visualizer
from brain_pipeline.utils import print_section


def main():
    print_section("STEP 3: CREATING VISUALIZATIONS")
    
    # Load configuration
    config = Config()
    visualizer = Visualizer(config)
    
    # === [1/4] Load results ===
    print("\n[1/4] Loading results...")

    # ----- Load PIOP-2 training folds -----
    piop2_files = sorted(glob.glob('data/results/error_map_piop2_training_fold*.csv'))
    if not piop2_files:
        raise FileNotFoundError(
            "No PIOP-2 training fold files found (expected pattern: data/results/error_map_piop2_training_fold*.csv)"
        )

    df_piop2_list = []
    for f in piop2_files:
        df = pd.read_csv(f)
        df["fold"] = Path(f).stem  # add fold label
        df_piop2_list.append(df)
    error_map_train_all = pd.concat(df_piop2_list, ignore_index=True)
    print(f"Loaded {len(piop2_files)} PIOP-2 training fold files ({len(error_map_train_all)} rows).")

    # ✅ Aggregate by region (average across folds)
    error_map_train = (
        error_map_train_all.groupby("region_name", as_index=False)
        .agg({"misclassification_rate": "mean"})
        .sort_values("misclassification_rate", ascending=False)
    )
    print(f"After aggregation: {len(error_map_train)} unique regions (expected ≈232).")

    # ----- Load PIOP-1 test folds -----
    piop1_files = sorted(glob.glob('data/results/error_map_piop1_fold*.csv'))
    if not piop1_files:
        raise FileNotFoundError(
            "No PIOP-1 fold files found (expected pattern: data/results/error_map_piop1_fold*.csv)"
        )

    df_piop1_list = []
    for f in piop1_files:
        df = pd.read_csv(f)
        df["fold"] = Path(f).stem
        df_piop1_list.append(df)
    error_map_test_all = pd.concat(df_piop1_list, ignore_index=True)
    print(f"Loaded {len(piop1_files)} PIOP-1 test fold files ({len(error_map_test_all)} rows).")

    # ✅ Aggregate by region (average across folds)
    error_map_test = (
        error_map_test_all.groupby("region_name", as_index=False)
        .agg({"misclassification_rate": "mean"})
        .sort_values("misclassification_rate", ascending=False)
    )
    print(f"After aggregation: {len(error_map_test)} unique regions (expected ≈232).")

    # ----- Load comparison fold files -----
    comparison_files = sorted(glob.glob('data/results/error_comparison_rest_vs_task_fold*.csv'))
    if not comparison_files:
        raise FileNotFoundError(
            "No comparison CSVs found (expected pattern: data/results/error_comparison_rest_vs_task_fold*.csv)"
        )

    df_comparison_list = [pd.read_csv(f) for f in comparison_files]
    comparison_all = pd.concat(df_comparison_list, ignore_index=True)
    print(f"Loaded {len(comparison_files)} comparison fold files ({len(comparison_all)} rows).")

    # ✅ Aggregate comparison (mean across folds per region)
    comparison = (
        comparison_all.groupby("region_name", as_index=False)
        .agg({"error_increase": "mean"})
        .sort_values("error_increase", ascending=False)
    )
    print(f"After aggregation: {len(comparison)} unique regions (expected ≈232).")
    
    # === [2/4] Create error map visualizations ===
    print("\n[2/4] Creating error map visualizations...")
    
    # Training error map (PIOP-2)
    fig1 = visualizer.plot_error_map(error_map_train, "PIOP-2 Training Error Map")
    visualizer.save_figure(fig1, 'error_map_piop2.png')
    
    # Test error map (PIOP-1)
    fig2 = visualizer.plot_error_map(error_map_test, "PIOP-1 Task Error Map")
    visualizer.save_figure(fig2, 'error_map_piop1.png')
    
    # === [3/4] Comparison visualization ===
    print("\n[3/4] Creating comparison visualizations...")
    fig3 = create_comparison_plot(error_map_train, error_map_test, comparison, config)
    visualizer.save_figure(fig3, 'comparison_rest_vs_task.png')
    
    # === [4/4] Network-level analysis ===
    print("\n[4/4] Creating network-level analysis...")
    fig4 = create_network_analysis(error_map_test, config)
    visualizer.save_figure(fig4, 'network_analysis.png')
    
    # === Summary ===
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(" error_map_piop2.png")
    print(" error_map_piop1.png")
    print(" comparison_rest_vs_task.png")
    print(" network_analysis.png")


# ==============================================================
#  COMPARISON VISUALIZATION FUNCTION
# ==============================================================

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


# ==============================================================
#  NETWORK-LEVEL ANALYSIS FUNCTION
# ==============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_network_analysis(error_map, config):
    """
    Create cortical–subcortical network-level analysis visualization
    Compatible with Schaefer-200x17 (cortex) and Tian S2x3T (subcortex)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- 1️⃣ Parse atlas and networks ---
    def parse_atlas_and_networks(region_name):
        # --- Cortical: Schaefer 200x17 ---
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

        # --- Subcortical: Tian S2x3T ---
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

    # --- Apply parsing to all regions ---
    error_map[['atlas', 'network_7', 'network_17']] = error_map['region_name'].apply(
        lambda r: pd.Series(parse_atlas_and_networks(r))
    )

    # --- Hemisphere detection ---
    error_map['hemisphere'] = error_map['region_name'].apply(
        lambda x: 'Left' if x.startswith('LH_') or x.endswith('-lh')
        else 'Right' if x.startswith('RH_') or x.endswith('-rh')
        else 'Unknown'
    )

    # --- 2️⃣ Network version control ---
    network_version = config.get("network_version", "7")  # can be "7" or "17"
    if network_version == "7":
        network_column = "network_7"
        title_suffix = " (7-Network Schaefer)"
    else:
        network_column = "network_17"
        title_suffix = " (17-Network Schaefer)"

    # --- 3️⃣ Plot 1: Network-level errors ---
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

    # --- 4️⃣ Plot 2: Hemisphere comparison ---
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

    # --- 5️⃣ Plot 3: Atlas comparison (Schaefer vs Tian) ---
    ax3 = axes[1, 0]
    sns.violinplot(
        data=error_map, y='atlas', x='misclassification_rate', hue = 'atlas',
        ax=ax3, palette='coolwarm', legend=False
    )
    ax3.set_xlabel('Misclassification Rate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Atlas (Cortex vs Subcortex)', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution by Atlas', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # --- 6️⃣ Plot 4: Atlas-level summary ---
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
    fontsize=13,                   # ← larger font
    family='monospace',
    verticalalignment='center',
    fontweight='bold',
    linespacing=1.5,               # ← more spacing between lines
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    main()