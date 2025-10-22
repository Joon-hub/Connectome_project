import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


class ConnectivityProcessor:
    """Process connectivity data and extract brain regions."""

    def __init__(self):
        self.region_list: List[str] = []
        self.region_to_idx: Dict[str, int] = {}
        self.n_regions: int = 0

    def extract_regions(self, connection_columns: List[str]) -> Tuple[List[str], Dict[str, int], int]:
        """Extract unique brain regions from connection column names."""
        unique_regions = set()
        for col in connection_columns:
            regions = col.split("~")
            if len(regions) == 2:
                unique_regions.update(regions)

        self.region_list = list(unique_regions)
        self.n_regions = len(self.region_list)
        self.region_to_idx = {region: idx for idx, region in enumerate(self.region_list)}

        print(f"Extracted {self.n_regions} unique brain regions.")
        return self.region_list, self.region_to_idx, self.n_regions

    def reconstruct_matrix(self, row_data: np.ndarray, connection_columns: List[str]) -> np.ndarray:
        """Reconstruct full connectivity matrix (symmetric) from a subject's row data.

        Parameters
        ----------
        row_data : np.ndarray
            Flattened connectivity values for one subject.
        connection_columns : List[str]
            Names of the connection columns in the same order as row_data.

        Returns
        -------
        np.ndarray
            A symmetric n_regions × n_regions connectivity matrix.
        """
        if self.n_regions == 0 or not self.region_to_idx:
            raise ValueError("Regions not initialized. Run extract_regions() first.")

        matrix = np.zeros((self.n_regions, self.n_regions), dtype=float)

        for col_name, value in zip(connection_columns, row_data):
            regions = col_name.split("~")
            if len(regions) == 2:
                try:
                    idx1 = self.region_to_idx[regions[0]]
                    idx2 = self.region_to_idx[regions[1]]
                    matrix[idx1, idx2] = value
                    matrix[idx2, idx1] = value
                except KeyError as e:
                    raise KeyError(f"Region not found in mapping: {e}")

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def create_dataset(
        self, df: pd.DataFrame, connection_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create region-level dataset (X, y, subjects) from connectivity data.

        For each subject, reconstructs their connectivity matrix and extracts
        the connectivity pattern of each region (excluding self-connection).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - X: (samples × features) array of region connectivity patterns
            - y: (samples,) array of region indices
            - subjects: (samples,) array of subject IDs
        """
        if self.n_regions == 0:
            raise ValueError("Regions not initialized. Run extract_regions() first.")

        X_list, y_list, subject_list = [], [], []

        for i in range(df.shape[0]):
            subject_id = df.iloc[i, 0]
            connectivity_values = df.iloc[i, 1:].to_numpy(dtype=float)

            conn_matrix = self.reconstruct_matrix(connectivity_values, connection_columns)

            # For each region, take its connectivity profile (excluding self)
            for region_idx in range(self.n_regions):
                connectivity_pattern = np.delete(conn_matrix[region_idx, :], region_idx)
                X_list.append(connectivity_pattern)
                y_list.append(region_idx)
                subject_list.append(subject_id)

        X = np.array(X_list)
        y = np.array(y_list, dtype=int)
        subjects = np.array(subject_list)

        print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features per region.")
        return X, y, subjects

    # ------------------------------------------------------
    # Additional helper methods for saving / loading metadata
    # ------------------------------------------------------

    def save_region_list(self, region_list: List[str], filepath: str) -> None:
        """Save the extracted region list to a CSV file."""
        pd.DataFrame(region_list, columns=["region_name"]).to_csv(filepath, index=False)
        print(f"Saved region list to {filepath}")

    def save_connection_columns(self, connection_columns: List[str], filepath: str) -> None:
        """Save connection column names to a CSV file."""
        pd.DataFrame(connection_columns, columns=["connection_name"]).to_csv(filepath, index=False)
        print(f"Saved connection columns to {filepath}")
