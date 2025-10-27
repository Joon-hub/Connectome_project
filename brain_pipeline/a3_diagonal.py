import numpy as np
from sklearn.impute import KNNImputer
from brain_pipeline.a0_config import Config


class DiagonalImputer:
    """
    Impute diagonal elements of a connectivity (or covariance-like) matrix
    using different strategies: zero, random, mean, one, or knn.
    """
    def __init__(self,matrix):
        # ensure input is numeric float type
        self.matrix = matrix.astype(float)
        self.config = Config()
        
        # load parameters from YAML via Config helper
        self.strategy = self.config.get('preprocessing','diagonal_imputation_settings', 'strategy', fallback='mean')
        self.knn_neighbors = self.config.get('preprocessing','diagonal_imputation_settings', 'knn_neighbors', fallback=3)
        self.seed = self.config.get('preprocessing','diagonal_imputation_settings', 'random_seed', fallback=42)
        

    
    def replace_diagonal_with_zero(self):
        m = self.matrix.copy()
        np.fill_diagonal(m, 0)
        return m

    def replace_diagonal_with_random(self):
        m = self.matrix.copy()
        np.random.seed(self.seed)
        np.fill_diagonal(m, np.random.uniform(-1, 1, m.shape[0]))
        return m

    def replace_diagonal_with_mean(self):
        m = self.matrix.copy()
        row_sums = m.sum(axis=1) - np.diag(m)
        row_means = row_sums / (m.shape[1] - 1)
        np.fill_diagonal(m, row_means)
        return m

    def replace_diagonal_with_one(self):
        m = self.matrix.copy()
        np.fill_diagonal(m, 1.0)
        return m

    def replace_diagonal_with_knn(self):
        m = self.matrix.copy()
        np.fill_diagonal(m, np.nan)

        imputer = KNNImputer(n_neighbors=self.knn_neighbors, metric='nan_euclidean')
        m_imputed = imputer.fit_transform(m)

        # Optional: re-symmetrize
        m_imputed = 0.5 * (m_imputed + m_imputed.T)
        return m_imputed

    def impute_diagonal(self, strategy=None):
        """Main imputation method controlled by config.yaml or manual override"""
        chosen_strategy = strategy or self.strategy

        if chosen_strategy == "zero":
            return self.replace_diagonal_with_zero()
        elif chosen_strategy == "random":
            return self.replace_diagonal_with_random()
        elif chosen_strategy == "mean":
            return self.replace_diagonal_with_mean()
        elif chosen_strategy == "one":
            return self.replace_diagonal_with_one()
        elif chosen_strategy == "knn":
            return self.replace_diagonal_with_knn()
        else:
            raise ValueError(f"Unknown strategy: {chosen_strategy}")


# Utility function for pipeline integration
def impute_connectivity_diagonal(matrix: np.ndarray, config: Config) -> np.ndarray:
    """Wrapper for pipeline use: imputes the diagonal of a given matrix."""
    imputer = DiagonalImputer(matrix, config)
    return imputer.impute_diagonal()

