import numpy as np
import yaml
from sklearn.impute import KNNImputer

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class DiagonalImputer:
    """Impute diagonal elements of a connectivity matrix with configurable strategies."""
    def __init__(self, matrix):
        self.matrix = matrix.astype(float)
        self.config = load_config()
        
        settings = (
            self.config.get('preprocessing', {})
            .get('diagonal_imputation_settings', {})
        )
        
        self.strategy = settings.get('strategy', 'mean')
        self.knn_neighbors = settings.get('knn_neighbors', 3)
        self.seed = settings.get('random_seed', 42)

    def replace_diagonal_with_zero(self):
        m = self.matrix.copy()
        np.fill_diagonal(m, 0)
        return m

    def replace_diagonal_with_random(self):
        m = self.matrix.copy()
        rng = np.random.default_rng(self.seed)
        np.fill_diagonal(m, rng.uniform(-1, 1, m.shape[0]))
        return m

    def replace_diagonal_with_mean(self):
        m = self.matrix.copy()
        if m.shape[0] == 1:
            np.fill_diagonal(m, 0)
            return m
        row_sums = np.sum(m, axis=1) - np.diag(m)
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
        return m_imputed

    def impute_diagonal(self, strategy=None):
        chosen_strategy = strategy or self.strategy
        strategies = {
            'zero': self.replace_diagonal_with_zero,
            'random': self.replace_diagonal_with_random,
            'mean': self.replace_diagonal_with_mean,
            'one': self.replace_diagonal_with_one,
            'knn': self.replace_diagonal_with_knn
        }
        if chosen_strategy not in strategies:
            raise ValueError(f"Unknown strategy: {chosen_strategy}")
        return strategies[chosen_strategy]()

def impute_connectivity_diagonal(matrix: np.ndarray) -> np.ndarray:
    imputer = DiagonalImputer(matrix)
    return imputer.impute_diagonal()
