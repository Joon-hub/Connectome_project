
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

class Visualizer:
    '''Create visualizations for results'''
    
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.get('data', 'results_dir'))
        self.dpi = config.get('visualization', 'dpi')
        self.color_scheme = config.get('visualization', 'color_scheme')
    
    def plot_error_map(self, error_map: pd.DataFrame, title: str = "Error Map"):
        '''Create comprehensive error map visualization'''
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: All regions
        ax1 = axes[0, 0]
        colors = plt.cm.get_cmap(self.color_scheme)(
            error_map['misclassification_rate'] / error_map['misclassification_rate'].max()
        )
        ax1.bar(range(len(error_map)), error_map['misclassification_rate'].values, 
                color=colors, edgecolor='black', linewidth=0.5)
        ax1.axhline(error_map['misclassification_rate'].mean(), color='blue', 
                   linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('Region Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Misclassification Rate', fontsize=12, fontweight='bold')
        ax1.set_title(f'{title} - All Regions', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Top 20 worst
        ax2 = axes[0, 1]
        top_20 = error_map.head(20)
        ax2.barh(range(20), top_20['misclassification_rate'].values, 
                color='red', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(20))
        ax2.set_yticklabels([name[:35] for name in top_20['region_name']], fontsize=9)
        ax2.set_xlabel('Misclassification Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Top 20 Misclassified Regions', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        # Plot 3: Distribution
        ax3 = axes[1, 0]
        ax3.hist(error_map['misclassification_rate'], bins=50, 
                alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(error_map['misclassification_rate'].mean(), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax3.set_xlabel('Misclassification Rate', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Regions', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution of Error Rates', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary stats
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats = f"""
        SUMMARY STATISTICS
        ==================
        
        Total Regions: {len(error_map)}
        Mean Error: {error_map['misclassification_rate'].mean():.4f}
        Median Error: {error_map['misclassification_rate'].median():.4f}
        Std Error: {error_map['misclassification_rate'].std():.4f}
        Min Error: {error_map['misclassification_rate'].min():.4f}
        Max Error: {error_map['misclassification_rate'].max():.4f}
        """
        ax4.text(0.1, 0.5, stats, fontsize=12, family='monospace',
                verticalalignment='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename: str):
        '''Save figure to results directory'''
        filepath = self.results_dir / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
        plt.close(fig)