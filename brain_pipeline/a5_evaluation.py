import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import Dict


class ModelEvaluator:
    """Evaluate model performance and create error maps."""

    def __init__(self, config, region_list):
        self.config = config
        self.region_list = region_list
        self.n_regions = len(region_list)

    def calculate_error_map(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Calculate per-region misclassification rates."""
        misclassification_rate = np.zeros(self.n_regions)
        n_samples = np.zeros(self.n_regions)

        for region_idx in range(self.n_regions):
            region_mask = y_true == region_idx
            if region_mask.any():
                y_true_region = y_true[region_mask]
                y_pred_region = y_pred[region_mask]
                misclassification_rate[region_idx] = 1 - accuracy_score(y_true_region, y_pred_region)
                n_samples[region_idx] = region_mask.sum()

        error_map_df = pd.DataFrame({
            "region_index": range(self.n_regions),
            "region_name": self.region_list,
            "misclassification_rate": np.round(misclassification_rate, 4),
            "n_samples": n_samples.astype(int),
        }).sort_values("misclassification_rate", ascending=False)

        return error_map_df

    def compare_datasets(
        self, error_map_train: pd.DataFrame, error_map_test: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare error rates between training and test datasets."""
        comparison = pd.merge(
            error_map_train[["region_name", "misclassification_rate"]],
            error_map_test[["region_name", "misclassification_rate"]],
            on="region_name",
            suffixes=("_train", "_test"),
        )
        comparison["error_increase"] = (
            comparison["misclassification_rate_test"]
            - comparison["misclassification_rate_train"]
        )
        comparison = comparison.sort_values("error_increase", ascending=False)
        return comparison

    def get_summary_stats(self, error_map: pd.DataFrame) -> Dict[str, float]:
        """Get summary statistics for misclassification rates."""
        
        # Define thresholds
        high_threshold = self.config.get("thresholds", "high_error", fallback=0.4)
        low_threshold = self.config.get("thresholds", "low_error", fallback=0.1)
        
       # Calculate summary statistics
        return {
            "mean_error": error_map["misclassification_rate"].mean(),
            "median_error": error_map["misclassification_rate"].median(),
            "std_error": error_map["misclassification_rate"].std(),
            "min_error": error_map["misclassification_rate"].min(),
            "max_error": error_map["misclassification_rate"].max(),
            "n_high_error": (error_map["misclassification_rate"] > high_threshold).sum(),
            "n_low_error": (error_map["misclassification_rate"] < low_threshold).sum(),
        }

    def save_results(self, error_map: pd.DataFrame, filename: str) -> None:
        """Save error map to CSV in results directory."""
        results_dir = Path(self.config.get("data", "results_dir")).resolve()
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / filename
        error_map.to_csv(filepath, index=False)
        print(f"Saved results: {filepath}")
