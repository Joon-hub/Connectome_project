import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


class DataLoader:
    """Load and manage brain connectivity data."""

    def __init__(self, config: object):
        self.config = config
        self.data_dir = Path(config.get("data", "raw_dir")).resolve()

    def load_piop2(self) -> pd.DataFrame:
        """Load PIOP-2 resting-state connectivity data."""
        filename = self.config.get("data", "piop2_file")
        filepath = self.data_dir / filename
        self._check_file_exists(filepath)
        df = pd.read_csv(filepath)
        print(f"Loaded PIOP-2 data from {filepath.name} with shape {df.shape}")
        return df

    def load_piop1(self) -> pd.DataFrame:
        """Load PIOP-1 gender task connectivity data."""
        filename = self.config.get("data", "piop1_file")
        filepath = self.data_dir / filename
        self._check_file_exists(filepath)
        df = pd.read_csv(filepath)
        print(f"Loaded PIOP-1 data from {filepath.name} with shape {df.shape}")
        return df

    def get_connection_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract all connection feature columns (assumes first column is subject ID)."""
        return df.columns[1:].tolist()

    def get_subjects(self, df: pd.DataFrame) -> np.ndarray:
        """Extract subject IDs (first column)."""
        return df.iloc[:, 0].to_numpy()

    @staticmethod
    def _check_file_exists(filepath: Path) -> None:
        """Ensure the specified file exists."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
