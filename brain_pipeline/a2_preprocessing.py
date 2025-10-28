"""
Enhanced Preprocessing Module for Brain Connectivity Data
==========================================================
Includes extraction of brain regions from connection columns,
reconstruction of connectivity matrices, and dataset creation
with advanced diagonal imputation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import sys
from pathlib import Path

# Import the enhanced diagonal imputation module
# Adjust path as needed based on your project structure
sys.path.append(str(Path(__file__).parent))
from brain_pipeline_a3_diagonal_enhanced import impute_connectivity_diagonal


class ConnectivityProcessor:
    """Process connectivity data and extract brain regions."""

    def __init__(self, config):
        self.config = config
        self.region_list: List[str] = []
        self.region_to_idx: Dict[str, int] = {}
        self.n_regions: int = 0

    def extract_regions(self, connection_columns: List[str]) -> Tuple[List[str], Dict[str, int], int]:
        """Extract unique brain regions from connection column names."""
        
        if not connection_columns:
            raise ValueError("Connection columns list is empty.")
        
        # Use list to preserve insertion order
        unique_regions = []
        seen_regions = set()
        
        for col in connection_columns:
            if "~" not in col:
                raise ValueError(f"Invalid connection column format: {col}")
            
            regions = col.split("~")
            if len(regions) != 2:
                raise ValueError(f"Connection column does not represent a pair: {col}")
            
            # Add regions in order of first appearance
            for region in regions:
                if region not in seen_regions:
                    seen_regions.add(region)
                    unique_regions.append(region)

        # Store in order of first appearance (no sorting)
        self.region_list = unique_regions
        self.n_regions = len(self.region_list)
        self.region_to_idx = {region: idx for idx, region in enumerate(self.region_list)}

        # Debug print
        print(f"Extracted {self.n_regions} unique brain regions.")

        # Return the extracted information
        return self.region_list, self.region_to_idx, self.n_regions

    def reconstruct_matrix(self, row_data: np.ndarray, connection_columns: List[str]) -> np.ndarray:
        """Reconstruct full connectivity matrix (symmetric) from a subject's row data."""
        
        if self.n_regions == 0:
            raise ValueError("Regions have not been extracted. Run extract_regions() first.")
        
        if len(row_data) != len(connection_columns):
            raise ValueError(
                f"Dimension mismatch: row_data has {len(row_data)} elements, "
                f"but connection_columns has {len(connection_columns)} elements."
            )
        
        # Initialize empty connectivity matrix
        matrix = np.zeros((self.n_regions, self.n_regions), dtype=float)
        
        # Set diagonal to 1.0 (self-connections) first
        np.fill_diagonal(matrix, 1.0)
        
        # Fill matrix based on connection columns and row data
        for col, value in zip(connection_columns, row_data):
            regions = col.split("~")
            
            # Validate regions exist in mapping
            if regions[0] not in self.region_to_idx:
                raise ValueError(f"Region {regions[0]} not found in region mapping.")
            if regions[1] not in self.region_to_idx:
                raise ValueError(f"Region {regions[1]} not found in region mapping.")
            
            # Get indices for the two regions
            idx1 = self.region_to_idx[regions[0]]
            idx2 = self.region_to_idx[regions[1]]
            
            # Fill both [idx1, idx2] and [idx2, idx1] to ensure symmetry
            if idx1 != idx2:
                matrix[idx1, idx2] = value
                matrix[idx2, idx1] = value

        return matrix
    
    def create_dataset(self, df: pd.DataFrame, connection_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create dataset of connectivity matrices and labels from DataFrame,
        with enhanced diagonal imputation that considers network structure.
        
        Key enhancement: Passes region_list to diagonal imputation to enable
        network-aware and spatially-aware strategies.
        """

        if self.n_regions == 0:
            raise ValueError("Regions have not been extracted. Run extract_regions() first.")
        
        if df.shape[0] == 0:
            raise ValueError("Input DataFrame is empty.")

        expected_cols = len(connection_columns) + 1
        if df.shape[1] != expected_cols:
            raise ValueError(
                f"Dimension mismatch: DataFrame has {df.shape[1]} columns, expected {expected_cols}."
            )
        
        # Get imputation strategy from config
        settings = (
            self.config.get('preprocessing', {})
            .get('diagonal_imputation_settings', {})
        )
        strategy = settings.get('strategy', 'mean')
        
        print(f"\n{'='*70}")
        print(f"Creating dataset with diagonal imputation strategy: {strategy}")
        print(f"{'='*70}")
        
        X_list, y_list, subject_list = [], [], []

        for i in range(df.shape[0]):
            subject_id = df.iloc[i, 0]
            connectivity_values = df.iloc[i, 1:].to_numpy(dtype=float)
            
            # Step 1: Reconstruct matrix
            matrix = self.reconstruct_matrix(connectivity_values, connection_columns)
            
            # Step 2: Enhanced diagonal imputation with region information
            # This enables network-aware and spatial strategies
            matrix = impute_connectivity_diagonal(
                matrix,
                region_list=self.region_list,  # KEY: Pass region names
                strategy=None,  # Use config default
                config_path="config.yaml"
            )
            
            # Step 3: Flatten rows for each region
            for region_idx in range(self.n_regions):
                X_list.append(matrix[region_idx, :])
                y_list.append(region_idx)
                subject_list.append(subject_id)

        X = np.array(X_list)
        y = np.array(y_list)
        subjects = np.array(subject_list)

        print(f"\n✓ Created dataset with {X.shape[0]} samples ({df.shape[0]} subjects × {self.n_regions} regions)")
        print(f"✓ Diagonal imputation strategy used: {strategy}")
        print(f"{'='*70}\n")
        
        return X, y, subjects
    
    def save_region_list(self, filepath: str) -> None:
        """Save the extracted region list to a CSV file."""
        if not self.region_list:
            raise ValueError("No regions to save. Run extract_regions() first.")
        
        pd.DataFrame(self.region_list, columns=["region_name"]).to_csv(filepath, index=False)
        print(f"Saved {len(self.region_list)} regions to {filepath}")

    def save_connection_columns(self, connection_columns: List[str], filepath: str) -> None:
        """Save connection column names to a CSV file."""
        if not connection_columns:
            raise ValueError("connection_columns cannot be empty")
        
        pd.DataFrame(connection_columns, columns=["connection_name"]).to_csv(filepath, index=False)
        print(f"Saved {len(connection_columns)} connection columns to {filepath}")


# ============================================================
# UTILITY: Compare diagonal imputation strategies
# ============================================================

def compare_diagonal_strategies(connectivity_matrix: np.ndarray,
                               region_list: List[str],
                               strategies: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Compare different diagonal imputation strategies on the same matrix.
    
    Args:
        connectivity_matrix: Input connectivity matrix
        region_list: List of region names
        strategies: List of strategy names to compare
        
    Returns:
        Dictionary mapping strategy name to imputed matrix
    """
    if strategies is None:
        strategies = ['zero', 'one', 'mean', 'knn', 'network_mean', 'spatial_mean', 'hybrid']
    
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING DIAGONAL IMPUTATION STRATEGIES")
    print("="*70)
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        try:
            imputed = impute_connectivity_diagonal(
                connectivity_matrix.copy(),
                region_list=region_list,
                strategy=strategy
            )
            results[strategy] = imputed
            
            # Quick stats
            diagonal_values = np.diag(imputed)
            print(f"  Diagonal mean: {diagonal_values.mean():.4f}")
            print(f"  Diagonal std:  {diagonal_values.std():.4f}")
            print(f"  Diagonal min:  {diagonal_values.min():.4f}")
            print(f"  Diagonal max:  {diagonal_values.max():.4f}")
            
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            results[strategy] = None
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")
    
    return results