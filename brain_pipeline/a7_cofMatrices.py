"""
Confusion Matrix Analysis for Brain Region Classification
==========================================================
Comprehensive confusion matrix generation and visualization at multiple hierarchical levels:
1. Region-wise (232x232)
2. Schaefer 17-network 
3. Schaefer 7-network
4. Tian subcortical network
5. Atlas-level (Cortical vs Subcortical)

Uses modern visualization techniques from recent neuroimaging literature.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ConfusionMatrixAnalyzer:
    """
    Analyze classification performance at multiple hierarchical levels.
    
    Attributes:
        region_list: List of all 232 region names
        y_true: True labels (region indices)
        y_pred: Predicted labels (region indices)
        config: Configuration object
    """
    
    def __init__(self, region_list: List[str], y_true: np.ndarray, 
                 y_pred: np.ndarray, config=None):
        """Initialize confusion matrix analyzer."""
        self.region_list = region_list
        self.y_true = y_true
        self.y_pred = y_pred
        self.config = config
        self.output_dir = Path("data/results/confusion_matrices")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse network information
        self.region_info = self._parse_all_regions()
        
    # ==========================================================================
    # REGION PARSING - Map regions to networks
    # ==========================================================================
    
    def _parse_all_regions(self) -> pd.DataFrame:
        """Parse all regions and assign network labels."""
        region_data = []
        
        for idx, region in enumerate(self.region_list):
            atlas, net7, net17, hemisphere = self._parse_single_region(region)
            region_data.append({
                'region_idx': idx,
                'region_name': region,
                'atlas': atlas,
                'network_7': net7,
                'network_17': net17,
                'hemisphere': hemisphere
            })
        
        return pd.DataFrame(region_data)
    
    def _parse_single_region(self, region_name: str) -> Tuple[str, str, str, str]:
        """Parse a single region name to extract atlas and network info."""
        
        # --- CORTICAL: Schaefer Atlas ---
        if region_name.startswith(('LH_', 'RH_')):
            atlas = 'Schaefer'
            hemisphere = 'Left' if region_name.startswith('LH_') else 'Right'
            
            # Schaefer 17-network mapping
            schaefer_17_map = {
                'VisCent': ('Visual', 'Visual Central'),
                'VisPeri': ('Visual', 'Visual Peripheral'),
                'SomMotA': ('Somatomotor', 'Somatomotor A'),
                'SomMotB': ('Somatomotor', 'Somatomotor B'),
                'DorsAttnA': ('Dorsal Attention', 'Dorsal Attention A'),
                'DorsAttnB': ('Dorsal Attention', 'Dorsal Attention B'),
                'SalVentAttnA': ('Salience/Ventral Attention', 'Salience/Ventral Attention A'),
                'SalVentAttnB': ('Salience/Ventral Attention', 'Salience/Ventral Attention B'),
                'LimbicA': ('Limbic', 'Limbic A'),
                'LimbicB': ('Limbic', 'Limbic B'),
                'ContA': ('Control', 'Control A'),
                'ContB': ('Control', 'Control B'),
                'ContC': ('Control', 'Control C'),
                'DefaultA': ('Default Mode', 'Default Mode A'),
                'DefaultB': ('Default Mode', 'Default Mode B'),
                'DefaultC': ('Default Mode', 'Default Mode C'),
                'TempPar': ('Temporal-Parietal', 'Temporal-Parietal')
            }
            
            for key, (net7, net17) in schaefer_17_map.items():
                if key in region_name:
                    return atlas, net7, net17, hemisphere
            
            return atlas, 'Other', 'Other', hemisphere
        
        # --- SUBCORTICAL: Tian Atlas ---
        else:
            atlas = 'Tian'
            hemisphere = 'Left' if region_name.endswith('-lh') else 'Right' if region_name.endswith('-rh') else 'Bilateral'
            
            # Tian subcortical network mapping
            tian_map = {
                'THA': 'Thalamus',
                'CAU': 'Caudate',
                'pCAU': 'Caudate',
                'aCAU': 'Caudate',
                'PUT': 'Putamen',
                'pPUT': 'Putamen',
                'aPUT': 'Putamen',
                'GP': 'Globus Pallidus',
                'pGP': 'Globus Pallidus',
                'aGP': 'Globus Pallidus',
                'HIP': 'Hippocampus',
                'pHIP': 'Hippocampus',
                'aHIP': 'Hippocampus',
                'AMY': 'Amygdala',
                'lAMY': 'Amygdala',
                'mAMY': 'Amygdala',
                'NAc': 'Nucleus Accumbens'
            }
            
            for key, structure in tian_map.items():
                if key in region_name:
                    return atlas, 'Subcortex', structure, hemisphere
            
            return atlas, 'Subcortex', 'Other Subcortical', hemisphere
    
    # ==========================================================================
    # CONFUSION MATRIX COMPUTATION
    # ==========================================================================
    
    def compute_region_confusion_matrix(self) -> np.ndarray:
        """Compute full 232x232 region-level confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred, 
                             labels=range(len(self.region_list)))
        return cm
    
    def compute_network_confusion_matrix(self, network_level: str = '17') -> Tuple[np.ndarray, List[str]]:
        """
        Compute confusion matrix at network level.
        
        Args:
            network_level: '7', '17', 'tian', or 'atlas'
        
        Returns:
            Confusion matrix and network labels
        """
        if network_level == '7':
            column = 'network_7'
        elif network_level == '17':
            column = 'network_17'
        elif network_level == 'tian':
            column = 'network_17'  # For Tian, network_17 contains subcortical structures
        elif network_level == 'atlas':
            column = 'atlas'
        else:
            raise ValueError(f"Unknown network level: {network_level}")
        
        # Map region indices to network labels
        true_networks = [self.region_info.loc[self.region_info['region_idx'] == idx, column].values[0] 
                        for idx in self.y_true]
        pred_networks = [self.region_info.loc[self.region_info['region_idx'] == idx, column].values[0] 
                        for idx in self.y_pred]
        
        # Get unique network labels
        unique_networks = sorted(set(true_networks + pred_networks))
        
        # Compute confusion matrix
        cm = confusion_matrix(true_networks, pred_networks, labels=unique_networks)
        
        return cm, unique_networks
    
    # ==========================================================================
    # VISUALIZATION - Modern Techniques
    # ==========================================================================
    
    def visualize_region_confusion_matrix(self, save_name: str = "region_confusion_matrix.png"):
        """
        Visualize full region-level confusion matrix with hierarchical clustering.
        Uses modern technique: hierarchical ordering by network + heatmap.
        """
        cm = self.compute_region_confusion_matrix()
        
        # Normalize by row (true labels) - shows where misclassifications go
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Sort regions by network for better visualization
        sorted_info = self.region_info.sort_values(['atlas', 'network_7', 'hemisphere'])
        sorted_indices = sorted_info['region_idx'].values
        
        # Reorder confusion matrix
        cm_sorted = cm_normalized[sorted_indices, :][:, sorted_indices]
        sorted_labels = sorted_info['region_name'].values
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.3],
                             hspace=0.3, wspace=0.3)
        
        # --- Main heatmap ---
        ax1 = fig.add_subplot(gs[0, :])
        
        # Use log scale for better visualization (many zeros)
        cm_log = np.log10(cm_sorted + 1e-6)
        
        im = ax1.imshow(cm_log, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax1.set_title('Region-Level Confusion Matrix (232×232)\nLog-scaled, sorted by network',
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add network boundaries
        network_boundaries = []
        current_network = None
        for i, idx in enumerate(sorted_indices):
            network = self.region_info.loc[self.region_info['region_idx'] == idx, 'network_7'].values[0]
            if network != current_network:
                network_boundaries.append(i)
                current_network = network
        
        for boundary in network_boundaries:
            ax1.axhline(boundary, color='white', linewidth=1.5, alpha=0.7)
            ax1.axvline(boundary, color='white', linewidth=1.5, alpha=0.7)
        
        # Sparse tick labels (every 20th region)
        tick_positions = range(0, len(sorted_labels), 20)
        ax1.set_xticks(tick_positions)
        ax1.set_yticks(tick_positions)
        ax1.set_xticklabels([sorted_labels[i][:30] for i in tick_positions], 
                           rotation=45, ha='right', fontsize=6)
        ax1.set_yticklabels([sorted_labels[i][:30] for i in tick_positions], fontsize=6)
        
        ax1.set_xlabel('Predicted Region', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Region', fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Log10(Probability)', fontsize=10, fontweight='bold')
        
        # --- Diagonal accuracy plot ---
        ax2 = fig.add_subplot(gs[1, 0])
        diagonal_acc = np.diag(cm_normalized)
        diagonal_sorted = diagonal_acc[sorted_indices]
        
        # Color by network
        network_colors = self._get_network_colors()
        colors = [network_colors.get(
            self.region_info.loc[self.region_info['region_idx'] == idx, 'network_7'].values[0], 
            'gray'
        ) for idx in sorted_indices]
        
        ax2.bar(range(len(diagonal_sorted)), diagonal_sorted, color=colors, 
               edgecolor='black', linewidth=0.3, alpha=0.8)
        ax2.axhline(diagonal_sorted.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {diagonal_sorted.mean():.3f}')
        ax2.set_xlabel('Region Index (sorted by network)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Recall (Diagonal Accuracy)', fontsize=10, fontweight='bold')
        ax2.set_title('Per-Region Classification Accuracy', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # --- Summary statistics ---
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        overall_acc = np.trace(cm) / np.sum(cm)
        mean_recall = diagonal_sorted.mean()
        
        stats_text = f"""
        REGION-LEVEL CLASSIFICATION SUMMARY
        ====================================
        
        Total Regions: {len(self.region_list)}
        Total Samples: {len(self.y_true):,}
        
        Overall Accuracy: {overall_acc:.4f}
        Mean Per-Region Recall: {mean_recall:.4f}
        
        Best Region: {sorted_labels[np.argmax(diagonal_sorted)]}
          → Recall: {diagonal_sorted.max():.4f}
        
        Worst Region: {sorted_labels[np.argmin(diagonal_sorted)]}
          → Recall: {diagonal_sorted.min():.4f}
        
        Regions with >80% recall: {(diagonal_sorted > 0.8).sum()}
        Regions with <50% recall: {(diagonal_sorted < 0.5).sum()}
        """
        
        ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved region-level confusion matrix: {save_name}")
    
    def visualize_network_confusion_matrix(self, network_level: str = '17',
                                          save_name: Optional[str] = None):
        """
        Visualize network-level confusion matrix.
        Modern technique: Annotated heatmap with precision/recall bars.
        
        Args:
            network_level: '7', '17', 'tian', or 'atlas'
        """
        if save_name is None:
            save_name = f"network_{network_level}_confusion_matrix.png"
        
        cm, network_labels = self.compute_network_confusion_matrix(network_level)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # --- 1. Confusion matrix heatmap ---
        ax1 = axes[0, 0]
        
        # Custom colormap
        im = ax1.imshow(cm_normalized, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(len(network_labels)):
            for j in range(len(network_labels)):
                value = cm_normalized[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=color, fontsize=8, fontweight='bold')
        
        ax1.set_xticks(range(len(network_labels)))
        ax1.set_yticks(range(len(network_labels)))
        ax1.set_xticklabels(network_labels, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels(network_labels, fontsize=9)
        ax1.set_xlabel('Predicted Network', fontsize=11, fontweight='bold')
        ax1.set_ylabel('True Network', fontsize=11, fontweight='bold')
        
        title_map = {
            '7': 'Schaefer 7-Network',
            '17': 'Schaefer 17-Network',
            'tian': 'Tian Subcortical',
            'atlas': 'Atlas-Level (Cortex vs Subcortex)'
        }
        ax1.set_title(f'{title_map.get(network_level, "Network")} Confusion Matrix',
                     fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Probability', fontsize=10, fontweight='bold')
        
        # --- 2. Precision and Recall bars ---
        ax2 = axes[0, 1]
        
        # Calculate metrics
        recall = np.diag(cm_normalized)
        precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        x = np.arange(len(network_labels))
        width = 0.25
        
        ax2.bar(x - width, recall, width, label='Recall', color='steelblue', 
               edgecolor='black', alpha=0.8)
        ax2.bar(x, precision, width, label='Precision', color='coral',
               edgecolor='black', alpha=0.8)
        ax2.bar(x + width, f1_score, width, label='F1-Score', color='seagreen',
               edgecolor='black', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(network_labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax2.set_title('Per-Network Performance Metrics', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        # Add mean lines
        ax2.axhline(recall.mean(), color='steelblue', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        ax2.axhline(precision.mean(), color='coral', linestyle='--',
                   linewidth=1.5, alpha=0.5)
        ax2.axhline(f1_score.mean(), color='seagreen', linestyle='--',
                   linewidth=1.5, alpha=0.5)
        
        # --- 3. Off-diagonal error analysis ---
        ax3 = axes[1, 0]
        
        # Sum of off-diagonal elements per row (where errors go)
        off_diag_errors = cm_normalized.copy()
        np.fill_diagonal(off_diag_errors, 0)
        error_distribution = off_diag_errors.sum(axis=1)
        
        colors_err = plt.cm.Reds(error_distribution / error_distribution.max())
        bars = ax3.barh(range(len(network_labels)), error_distribution,
                       color=colors_err, edgecolor='black', linewidth=1)
        ax3.set_yticks(range(len(network_labels)))
        ax3.set_yticklabels(network_labels, fontsize=9)
        ax3.set_xlabel('Total Misclassification Rate', fontsize=11, fontweight='bold')
        ax3.set_title('Networks with Highest Error Rates', fontsize=13, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # Annotate bars
        for i, (bar, val) in enumerate(zip(bars, error_distribution)):
            if val > 0.01:
                ax3.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=8)
        
        # --- 4. Summary statistics ---
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        overall_acc = np.trace(cm) / np.sum(cm)
        
        stats_text = f"""
        {title_map.get(network_level, "Network")} SUMMARY
        {'=' * 40}
        
        Number of Networks: {len(network_labels)}
        Total Samples: {len(self.y_true):,}
        
        Overall Accuracy: {overall_acc:.4f}
        
        Mean Precision: {precision.mean():.4f}
        Mean Recall: {recall.mean():.4f}
        Mean F1-Score: {f1_score.mean():.4f}
        
        Best Network (F1): {network_labels[np.argmax(f1_score)]}
          → F1: {f1_score.max():.4f}
        
        Worst Network (F1): {network_labels[np.argmin(f1_score)]}
          → F1: {f1_score.min():.4f}
        
        Networks with F1 > 0.8: {(f1_score > 0.8).sum()}
        Networks with F1 < 0.5: {(f1_score < 0.5).sum()}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {network_level}-network confusion matrix: {save_name}")
    
    def visualize_cross_network_errors(self, save_name: str = "cross_network_errors.png"):
        """
        Visualize which networks are most commonly confused with each other.
        Modern technique: Chord diagram style + hierarchical edge bundling concept.
        """
        cm, network_labels = self.compute_network_confusion_matrix('7')
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Remove diagonal (we only care about confusions)
        np.fill_diagonal(cm_normalized, 0)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # --- 1. Hierarchical heatmap of confusions ---
        ax1 = axes[0]
        
        im = ax1.imshow(cm_normalized, cmap='YlOrRd', vmin=0, vmax=cm_normalized.max(),
                       aspect='auto')
        
        # Annotate significant confusions only
        for i in range(len(network_labels)):
            for j in range(len(network_labels)):
                if i != j and cm_normalized[i, j] > 0.05:  # Only show >5% confusions
                    ax1.text(j, i, f'{cm_normalized[i, j]:.2f}', 
                            ha='center', va='center', color='black',
                            fontsize=9, fontweight='bold')
        
        ax1.set_xticks(range(len(network_labels)))
        ax1.set_yticks(range(len(network_labels)))
        ax1.set_xticklabels(network_labels, rotation=45, ha='right', fontsize=10)
        ax1.set_yticklabels(network_labels, fontsize=10)
        ax1.set_xlabel('Confused As', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Network', fontsize=12, fontweight='bold')
        ax1.set_title('Cross-Network Confusion Patterns', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Confusion Rate', fontsize=10, fontweight='bold')
        
        # --- 2. Top confusion pairs ---
        ax2 = axes[1]
        
        # Extract top confusion pairs
        confusion_pairs = []
        for i in range(len(network_labels)):
            for j in range(len(network_labels)):
                if i != j:
                    confusion_pairs.append({
                        'true': network_labels[i],
                        'predicted': network_labels[j],
                        'rate': cm_normalized[i, j]
                    })
        
        # Sort and take top 15
        confusion_pairs = sorted(confusion_pairs, key=lambda x: x['rate'], reverse=True)[:15]
        
        # Plot as horizontal bars
        y_pos = range(len(confusion_pairs))
        rates = [pair['rate'] for pair in confusion_pairs]
        labels = [f"{pair['true'][:15]} → {pair['predicted'][:15]}" for pair in confusion_pairs]
        
        colors = plt.cm.OrRd(np.linspace(0.4, 0.9, len(confusion_pairs)))
        bars = ax2.barh(y_pos, rates, color=colors, edgecolor='black', linewidth=1.5)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Confusion Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Top 15 Network Confusion Pairs', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Annotate bars
        for i, (bar, val) in enumerate(zip(bars, rates)):
            ax2.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved cross-network error analysis: {save_name}")
    
    def visualize_atlas_comparison(self, save_name: str = "atlas_comparison.png"):
        """
        Compare cortical (Schaefer) vs subcortical (Tian) classification performance.
        Modern technique: Violin plots + confusion between atlases.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Compute per-region accuracies
        cm_region = self.compute_region_confusion_matrix()
        cm_normalized = cm_region.astype('float') / cm_region.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        region_accuracies = np.diag(cm_normalized)
        
        # Add accuracies to region_info
        acc_df = self.region_info.copy()
        acc_df['accuracy'] = [region_accuracies[idx] for idx in acc_df['region_idx']]
        
        # --- 1. Violin plot: Atlas comparison ---
        ax1 = axes[0, 0]
        
        atlas_data = [
            acc_df[acc_df['atlas'] == 'Schaefer']['accuracy'].values,
            acc_df[acc_df['atlas'] == 'Tian']['accuracy'].values
        ]
        
        parts = ax1.violinplot(atlas_data, positions=[0, 1], widths=0.7,
                              showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['steelblue', 'coral']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Schaefer\n(Cortical)', 'Tian\n(Subcortical)'], fontsize=11)
        ax1.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Cortical vs Subcortical Performance', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Add statistics
        schaefer_mean = atlas_data[0].mean()
        tian_mean = atlas_data[1].mean()
        ax1.text(0, schaefer_mean + 0.05, f'μ={schaefer_mean:.3f}', 
                ha='center', fontweight='bold', fontsize=10)
        ax1.text(1, tian_mean + 0.05, f'μ={tian_mean:.3f}',
                ha='center', fontweight='bold', fontsize=10)
        
        # --- 2. Network-wise comparison (Schaefer 7) ---
        ax2 = axes[0, 1]
        
        schaefer_data = acc_df[acc_df['atlas'] == 'Schaefer']
        network_stats = schaefer_data.groupby('network_7')['accuracy'].agg(['mean', 'std', 'count'])
        network_stats = network_stats.sort_values('mean', ascending=True)
        
        colors_net = plt.cm.RdYlGn(network_stats['mean'] / network_stats['mean'].max())
        bars = ax2.barh(range(len(network_stats)), network_stats['mean'],
                       xerr=network_stats['std'], color=colors_net,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax2.set_yticks(range(len(network_stats)))
        ax2.set_yticklabels(network_stats.index, fontsize=10)
        ax2.set_xlabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Performance by Cortical Network', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Annotate
        for i, (idx, row) in enumerate(network_stats.iterrows()):
            ax2.text(row['mean'] + 0.02, i, f"{row['mean']:.3f}\n(n={int(row['count'])})",
                    va='center', fontsize=8, fontweight='bold')
        
        # --- 3. Subcortical structure comparison ---
        ax3 = axes[1, 0]
        
        tian_data = acc_df[acc_df['atlas'] == 'Tian']
        structure_stats = tian_data.groupby('network_17')['accuracy'].agg(['mean', 'std', 'count'])
        structure_stats = structure_stats.sort_values('mean', ascending=True)
        
        colors_struct = plt.cm.YlOrRd(structure_stats['mean'] / structure_stats['mean'].max())
        bars = ax3.barh(range(len(structure_stats)), structure_stats['mean'],
                       xerr=structure_stats['std'], color=colors_struct,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax3.set_yticks(range(len(structure_stats)))
        ax3.set_yticklabels(structure_stats.index, fontsize=10)
        ax3.set_xlabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Performance by Subcortical Structure', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Annotate
        for i, (idx, row) in enumerate(structure_stats.iterrows()):
            ax3.text(row['mean'] + 0.02, i, f"{row['mean']:.3f}\n(n={int(row['count'])})",
                    va='center', fontsize=8, fontweight='bold')
        
        # --- 4. Summary statistics table ---
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        ATLAS-LEVEL PERFORMANCE COMPARISON
        ====================================
        
        CORTICAL (Schaefer Atlas)
        -------------------------
        Regions: {(acc_df['atlas'] == 'Schaefer').sum()}
        Mean Accuracy: {schaefer_mean:.4f}
        Std Dev: {atlas_data[0].std():.4f}
        Min Accuracy: {atlas_data[0].min():.4f}
        Max Accuracy: {atlas_data[0].max():.4f}
        Regions >80% acc: {(atlas_data[0] > 0.8).sum()}
        
        SUBCORTICAL (Tian Atlas)
        -------------------------
        Regions: {(acc_df['atlas'] == 'Tian').sum()}
        Mean Accuracy: {tian_mean:.4f}
        Std Dev: {atlas_data[1].std():.4f}
        Min Accuracy: {atlas_data[1].min():.4f}
        Max Accuracy: {atlas_data[1].max():.4f}
        Regions >80% acc: {(atlas_data[1] > 0.8).sum()}
        
        DIFFERENCE
        ----------
        Cortical - Subcortical: {schaefer_mean - tian_mean:+.4f}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved atlas comparison: {save_name}")
    
    # ==========================================================================
    # COMPREHENSIVE REPORT
    # ==========================================================================
    
    def generate_classification_report(self, save_name: str = "classification_report.txt"):
        """Generate detailed text report with all metrics."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BRAIN REGION CLASSIFICATION - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall metrics
        cm_region = self.compute_region_confusion_matrix()
        overall_acc = np.trace(cm_region) / np.sum(cm_region)
        
        report_lines.append(f"Total Regions: {len(self.region_list)}")
        report_lines.append(f"Total Samples: {len(self.y_true):,}")
        report_lines.append(f"Overall Accuracy: {overall_acc:.4f}")
        report_lines.append("")
        
        # Network-level reports
        for level, name in [('7', 'Schaefer 7-Network'), 
                           ('17', 'Schaefer 17-Network'),
                           ('atlas', 'Atlas-Level')]:
            report_lines.append("-" * 80)
            report_lines.append(f"{name} PERFORMANCE")
            report_lines.append("-" * 80)
            
            cm, labels = self.compute_network_confusion_matrix(level)
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
            
            recall = np.diag(cm_norm)
            precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            for i, label in enumerate(labels):
                report_lines.append(f"\n{label}:")
                report_lines.append(f"  Recall:    {recall[i]:.4f}")
                report_lines.append(f"  Precision: {precision[i]:.4f}")
                report_lines.append(f"  F1-Score:  {f1[i]:.4f}")
            
            report_lines.append(f"\nMean Recall:    {recall.mean():.4f}")
            report_lines.append(f"Mean Precision: {precision.mean():.4f}")
            report_lines.append(f"Mean F1-Score:  {f1.mean():.4f}")
            report_lines.append("")
        
        # Save report
        report_path = self.output_dir / save_name
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Saved classification report: {save_name}")
    
    # ==========================================================================
    # HELPER FUNCTIONS
    # ==========================================================================
    
    def _get_network_colors(self) -> Dict[str, str]:
        """Get consistent color mapping for networks."""
        return {
            'Visual': '#7D3C98',
            'Somatomotor': '#2874A6',
            'Dorsal Attention': '#117A65',
            'Salience/Ventral Attention': '#D68910',
            'Limbic': '#BA4A00',
            'Control': '#A93226',
            'Default Mode': '#1F618D',
            'Temporal-Parietal': '#6C3483',
            'Subcortex': '#34495E',
            'Other': '#95A5A6'
        }
    
    def create_all_visualizations(self):
        """Generate all confusion matrix visualizations."""
        print("\n" + "="*70)
        print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
        print("="*70)
        
        print("\n[1/6] Creating region-level confusion matrix...")
        self.visualize_region_confusion_matrix()
        
        print("\n[2/6] Creating Schaefer 7-network confusion matrix...")
        self.visualize_network_confusion_matrix('7')
        
        print("\n[3/6] Creating Schaefer 17-network confusion matrix...")
        self.visualize_network_confusion_matrix('17')
        
        print("\n[4/6] Creating cross-network error analysis...")
        self.visualize_cross_network_errors()
        
        print("\n[5/6] Creating atlas-level comparison...")
        self.visualize_atlas_comparison()
        
        print("\n[6/6] Generating classification report...")
        self.generate_classification_report()
        
        print("\n" + "="*70)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir.absolute()}")


# =============================================================================
# STANDALONE SCRIPT USAGE
# =============================================================================

def main():
    """
    Example usage of ConfusionMatrixAnalyzer.
    Load predictions and generate all visualizations.
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from brain_pipeline import Config, DataLoader, ConnectivityProcessor, BrainRegionClassifier
    
    print("Loading data and model predictions...")
    
    # Load configuration
    config = Config()
    
    # Load data
    data_loader = DataLoader(config)
    df_piop2 = data_loader.load_piop2()
    connection_columns = data_loader.get_connection_columns(df_piop2)
    
    # Extract regions
    processor = ConnectivityProcessor(config)
    region_list, region_to_idx, n_regions = processor.extract_regions(connection_columns)
    
    # Create dataset
    X, y_true, subjects = processor.create_dataset(df_piop2, connection_columns)
    
    # Load trained model and get predictions
    classifier = BrainRegionClassifier(config)
    classifier.load("data/processed/trained_model_fold1.pkl")
    y_pred, _ = classifier.predict(X)
    
    # Create analyzer
    print("\nInitializing confusion matrix analyzer...")
    analyzer = ConfusionMatrixAnalyzer(region_list, y_true, y_pred, config)
    
    # Generate all visualizations
    analyzer.create_all_visualizations()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()