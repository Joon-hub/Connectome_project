"""
Enhanced Diagonal Imputation for Brain Connectivity Matrices
=============================================================
Advanced strategies for imputing diagonal elements that preserve
network structure and spatial relationships.

Strategies implemented:
1. Zero: Set diagonal to 0
2. One: Set diagonal to 1 (self-correlation)
3. Mean: Row-wise mean of off-diagonal elements
4. Random: Random values from uniform distribution
5. KNN: K-nearest neighbors imputation
6. Network-specific: Mean within same functional network (NEW)
7. Spatial neighborhood: Mean of spatially proximal regions (NEW)
8. Hybrid: Combines network and spatial information (NEW)
"""

import numpy as np
import yaml
from sklearn.impute import KNNImputer
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class NetworkParser:
    """Parse network membership from Schaefer and Tian region names."""
    
    # Schaefer 17-network mapping
    SCHAEFER_17_NETWORKS = {
        'VisCent': 'Visual_Central',
        'VisPeri': 'Visual_Peripheral',
        'SomMotA': 'Somatomotor_A',
        'SomMotB': 'Somatomotor_B',
        'DorsAttnA': 'DorsalAttention_A',
        'DorsAttnB': 'DorsalAttention_B',
        'SalVentAttnA': 'SalienceVentralAttention_A',
        'SalVentAttnB': 'SalienceVentralAttention_B',
        'LimbicA': 'Limbic_A',
        'LimbicB': 'Limbic_B',
        'ContA': 'Control_A',
        'ContB': 'Control_B',
        'ContC': 'Control_C',
        'DefaultA': 'DefaultMode_A',
        'DefaultB': 'DefaultMode_B',
        'DefaultC': 'DefaultMode_C',
        'TempPar': 'TemporalParietal'
    }
    
    # Schaefer 7-network mapping (coarser)
    SCHAEFER_7_NETWORKS = {
        'Vis': 'Visual',
        'SomMot': 'Somatomotor',
        'DorsAttn': 'DorsalAttention',
        'SalVentAttn': 'SalienceVentralAttention',
        'Limbic': 'Limbic',
        'Cont': 'Control',
        'Default': 'DefaultMode',
        'TempPar': 'TemporalParietal'
    }
    
    # Tian subcortical structures
    TIAN_SUBCORTICAL = {
        'THA': 'Thalamus',
        'CAU': 'Caudate',
        'PUT': 'Putamen',
        'GP': 'GlobusPallidus',
        'HIP': 'Hippocampus',
        'AMY': 'Amygdala',
        'NAc': 'NucleusAccumbens'
    }
    
    @classmethod
    def parse_network(cls, region_name: str, network_resolution: str = '17') -> str:
        """
        Parse network membership from region name.
        
        Args:
            region_name: Region name (e.g., 'LH_VisCent_ExStr_1' or 'lh-THA-1')
            network_resolution: '7' or '17' for Schaefer networks
            
        Returns:
            Network name string
        """
        # Cortical regions (Schaefer)
        if region_name.startswith(('LH_', 'RH_')):
            network_dict = (cls.SCHAEFER_17_NETWORKS if network_resolution == '17' 
                          else cls.SCHAEFER_7_NETWORKS)
            
            for key, network in network_dict.items():
                if key in region_name:
                    return network
            return 'Cortical_Unknown'
        
        # Subcortical regions (Tian)
        else:
            for key, structure in cls.TIAN_SUBCORTICAL.items():
                if key in region_name:
                    return f'Subcortex_{structure}'
            return 'Subcortical_Unknown'
    
    @classmethod
    def get_network_membership(cls, region_list: List[str], 
                               network_resolution: str = '17') -> np.ndarray:
        """
        Get network membership array for all regions.
        
        Args:
            region_list: List of region names
            network_resolution: '7' or '17' for Schaefer networks
            
        Returns:
            Array of network IDs (integers) for each region
        """
        networks = [cls.parse_network(r, network_resolution) for r in region_list]
        unique_networks = sorted(list(set(networks)))
        network_to_id = {net: idx for idx, net in enumerate(unique_networks)}
        network_ids = np.array([network_to_id[net] for net in networks])
        
        return network_ids, unique_networks


class SpatialNeighborhood:
    """Calculate spatial neighborhoods for brain regions."""
    
    # Approximate MNI coordinates for Schaefer centroids (example subset)
    # In practice, you would load these from the actual atlas
    # Format: region_index -> (x, y, z) in MNI space
    
    @staticmethod
    def load_region_coordinates(atlas_type: str = 'schaefer_200') -> Optional[np.ndarray]:
        """
        Load region centroid coordinates from atlas.
        
        Returns:
            Array of shape (n_regions, 3) with MNI coordinates
            or None if coordinates not available
        """
        # This is a placeholder - in real implementation, you would:
        # 1. Load NIfTI atlas file
        # 2. Extract centroids for each region
        # 3. Return as coordinate array
        
        warnings.warn(
            "Spatial coordinates not loaded. Using fallback spatial strategy. "
            "To enable full spatial imputation, provide atlas coordinates."
        )
        return None
    
    @staticmethod
    def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance matrix between region centroids.
        
        Args:
            coordinates: Array of shape (n_regions, 3) with MNI coordinates
            
        Returns:
            Distance matrix of shape (n_regions, n_regions)
        """
        n_regions = coordinates.shape[0]
        distances = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(n_regions):
                distances[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        return distances
    
    @staticmethod
    def get_spatial_neighbors(region_idx: int, distance_matrix: np.ndarray, 
                             radius_mm: float = 20.0, 
                             exclude_self: bool = True) -> np.ndarray:
        """
        Get indices of spatially neighboring regions within radius.
        
        Args:
            region_idx: Index of target region
            distance_matrix: Pairwise distance matrix
            radius_mm: Radius in millimeters
            exclude_self: Whether to exclude the region itself
            
        Returns:
            Array of neighbor indices
        """
        distances = distance_matrix[region_idx, :]
        neighbors = np.where(distances <= radius_mm)[0]
        
        if exclude_self:
            neighbors = neighbors[neighbors != region_idx]
        
        return neighbors


class DiagonalImputer:
    """
    Enhanced diagonal imputation with network-aware and spatial strategies.
    """
    
    def __init__(self, matrix: np.ndarray, region_list: Optional[List[str]] = None,
                 config_path: str = "config.yaml"):
        """
        Initialize imputer.
        
        Args:
            matrix: Connectivity matrix (n_regions, n_regions)
            region_list: List of region names (for network parsing)
            config_path: Path to configuration file
        """
        self.matrix = matrix.astype(float)
        self.n_regions = matrix.shape[0]
        self.region_list = region_list
        self.config = load_config(config_path)
        
        # Load settings
        settings = (
            self.config.get('preprocessing', {})
            .get('diagonal_imputation_settings', {})
        )
        
        self.strategy = settings.get('strategy', 'mean')
        self.knn_neighbors = settings.get('knn_neighbors', 3)
        self.seed = settings.get('random_seed', 42)
        self.network_resolution = settings.get('network_resolution', '17')
        self.spatial_radius = settings.get('spatial_radius_mm', 20.0)
        self.hybrid_weights = settings.get('hybrid_weights', {'network': 0.6, 'spatial': 0.4})
        
        # Initialize network and spatial information
        self.network_ids = None
        self.unique_networks = None
        self.distance_matrix = None
        
        if region_list is not None:
            self._initialize_network_info()
            self._initialize_spatial_info()
    
    def _initialize_network_info(self):
        """Parse network membership for all regions."""
        if self.region_list is None:
            warnings.warn("Region list not provided. Network-based strategies unavailable.")
            return
        
        self.network_ids, self.unique_networks = NetworkParser.get_network_membership(
            self.region_list, self.network_resolution
        )
        
        print(f"Parsed {len(self.unique_networks)} unique networks:")
        for i, net in enumerate(self.unique_networks):
            count = np.sum(self.network_ids == i)
            print(f"  {net}: {count} regions")
    
    def _initialize_spatial_info(self):
        """Load spatial coordinates and compute distance matrix."""
        coordinates = SpatialNeighborhood.load_region_coordinates()
        
        if coordinates is not None:
            self.distance_matrix = SpatialNeighborhood.compute_distance_matrix(coordinates)
            print(f"Loaded spatial coordinates for {coordinates.shape[0]} regions")
        else:
            print("Spatial coordinates not available. Using hemisphere-based approximation.")
            self._approximate_spatial_from_hemisphere()
    
    def _approximate_spatial_from_hemisphere(self):
        """
        Approximate spatial neighborhood using hemisphere information.
        Regions in same hemisphere are considered "closer".
        """
        if self.region_list is None:
            return
        
        # Create pseudo-distance matrix based on hemisphere
        self.distance_matrix = np.ones((self.n_regions, self.n_regions)) * 100.0
        
        for i, region_i in enumerate(self.region_list):
            hemisphere_i = self._get_hemisphere(region_i)
            
            for j, region_j in enumerate(self.region_list):
                hemisphere_j = self._get_hemisphere(region_j)
                
                if hemisphere_i == hemisphere_j and hemisphere_i != 'Unknown':
                    # Same hemisphere: closer (15mm equivalent)
                    self.distance_matrix[i, j] = 15.0
                elif hemisphere_i != hemisphere_j:
                    # Different hemisphere: far (100mm)
                    self.distance_matrix[i, j] = 100.0
                
                if i == j:
                    self.distance_matrix[i, j] = 0.0
    
    @staticmethod
    def _get_hemisphere(region_name: str) -> str:
        """Determine hemisphere from region name."""
        if region_name.startswith('LH_') or region_name.endswith('-lh'):
            return 'Left'
        elif region_name.startswith('RH_') or region_name.endswith('-rh'):
            return 'Right'
        else:
            return 'Unknown'
    
    # ============================================================
    # BASIC STRATEGIES (existing)
    # ============================================================
    
    def replace_diagonal_with_zero(self) -> np.ndarray:
        """Set diagonal to 0."""
        m = self.matrix.copy()
        np.fill_diagonal(m, 0)
        return m
    
    def replace_diagonal_with_one(self) -> np.ndarray:
        """Set diagonal to 1 (perfect self-correlation)."""
        m = self.matrix.copy()
        np.fill_diagonal(m, 1.0)
        return m
    
    def replace_diagonal_with_random(self) -> np.ndarray:
        """Replace diagonal with random values."""
        m = self.matrix.copy()
        rng = np.random.default_rng(self.seed)
        np.fill_diagonal(m, rng.uniform(-1, 1, m.shape[0]))
        return m
    
    def replace_diagonal_with_mean(self) -> np.ndarray:
        """Replace diagonal with row-wise mean of off-diagonal elements."""
        m = self.matrix.copy()
        
        if m.shape[0] == 1:
            np.fill_diagonal(m, 0)
            return m
        
        # Calculate row means excluding diagonal
        row_sums = np.sum(m, axis=1) - np.diag(m)
        row_means = row_sums / (m.shape[1] - 1)
        np.fill_diagonal(m, row_means)
        
        return m
    
    def replace_diagonal_with_knn(self) -> np.ndarray:
        """Use K-nearest neighbors imputation."""
        m = self.matrix.copy()
        np.fill_diagonal(m, np.nan)
        
        imputer = KNNImputer(n_neighbors=self.knn_neighbors, metric='nan_euclidean')
        m_imputed = imputer.fit_transform(m)
        
        return m_imputed
    
    # ============================================================
    # ADVANCED STRATEGIES (new)
    # ============================================================
    
    def replace_diagonal_with_network_mean(self) -> np.ndarray:
        """
        Replace diagonal with mean connectivity within same functional network.
        
        Rationale: Regions in the same network have similar connectivity profiles.
        The diagonal value should reflect typical within-network connectivity strength.
        """
        if self.network_ids is None:
            warnings.warn("Network information not available. Falling back to row mean.")
            return self.replace_diagonal_with_mean()
        
        m = self.matrix.copy()
        
        for region_idx in range(self.n_regions):
            region_network = self.network_ids[region_idx]
            
            # Find all regions in the same network (excluding self)
            same_network_mask = (self.network_ids == region_network)
            same_network_mask[region_idx] = False
            same_network_indices = np.where(same_network_mask)[0]
            
            if len(same_network_indices) > 0:
                # Mean connectivity to same-network regions
                network_connectivity = m[region_idx, same_network_indices]
                diagonal_value = np.mean(network_connectivity)
            else:
                # Fallback to row mean if no other regions in network
                row_sum = np.sum(m[region_idx, :]) - m[region_idx, region_idx]
                diagonal_value = row_sum / (self.n_regions - 1)
            
            m[region_idx, region_idx] = diagonal_value
        
        return m
    
    def replace_diagonal_with_spatial_mean(self) -> np.ndarray:
        """
        Replace diagonal with mean connectivity to spatially neighboring regions.
        
        Rationale: Spatially close regions often have correlated activity.
        The diagonal should reflect local connectivity strength.
        """
        if self.distance_matrix is None:
            warnings.warn("Spatial information not available. Falling back to row mean.")
            return self.replace_diagonal_with_mean()
        
        m = self.matrix.copy()
        
        for region_idx in range(self.n_regions):
            # Get spatial neighbors within radius
            neighbors = SpatialNeighborhood.get_spatial_neighbors(
                region_idx, self.distance_matrix, 
                radius_mm=self.spatial_radius,
                exclude_self=True
            )
            
            if len(neighbors) > 0:
                # Mean connectivity to spatial neighbors
                neighbor_connectivity = m[region_idx, neighbors]
                diagonal_value = np.mean(neighbor_connectivity)
            else:
                # Fallback to row mean if no neighbors
                row_sum = np.sum(m[region_idx, :]) - m[region_idx, region_idx]
                diagonal_value = row_sum / (self.n_regions - 1)
            
            m[region_idx, region_idx] = diagonal_value
        
        return m
    
    def replace_diagonal_with_hybrid(self) -> np.ndarray:
        """
        Hybrid strategy: weighted combination of network and spatial information.
        
        Rationale: Both network membership and spatial proximity matter for
        brain connectivity. This combines both sources of information.
        """
        if self.network_ids is None or self.distance_matrix is None:
            warnings.warn("Network or spatial info unavailable. Falling back to row mean.")
            return self.replace_diagonal_with_mean()
        
        m = self.matrix.copy()
        w_network = self.hybrid_weights['network']
        w_spatial = self.hybrid_weights['spatial']
        
        for region_idx in range(self.n_regions):
            # Network-based value
            region_network = self.network_ids[region_idx]
            same_network_mask = (self.network_ids == region_network)
            same_network_mask[region_idx] = False
            same_network_indices = np.where(same_network_mask)[0]
            
            if len(same_network_indices) > 0:
                network_value = np.mean(m[region_idx, same_network_indices])
            else:
                network_value = np.mean(m[region_idx, :])
            
            # Spatial-based value
            neighbors = SpatialNeighborhood.get_spatial_neighbors(
                region_idx, self.distance_matrix,
                radius_mm=self.spatial_radius,
                exclude_self=True
            )
            
            if len(neighbors) > 0:
                spatial_value = np.mean(m[region_idx, neighbors])
            else:
                spatial_value = np.mean(m[region_idx, :])
            
            # Weighted combination
            diagonal_value = w_network * network_value + w_spatial * spatial_value
            m[region_idx, region_idx] = diagonal_value
        
        return m
    
    def replace_diagonal_with_network_spatial_intersection(self) -> np.ndarray:
        """
        Use mean connectivity to regions that are BOTH in same network AND spatially close.
        
        Rationale: Most stringent criterion - regions must be both functionally
        and spatially related.
        """
        if self.network_ids is None or self.distance_matrix is None:
            warnings.warn("Network or spatial info unavailable. Falling back to row mean.")
            return self.replace_diagonal_with_mean()
        
        m = self.matrix.copy()
        
        for region_idx in range(self.n_regions):
            # Same network
            region_network = self.network_ids[region_idx]
            same_network_mask = (self.network_ids == region_network)
            same_network_mask[region_idx] = False
            
            # Spatial neighbors
            spatial_neighbors = SpatialNeighborhood.get_spatial_neighbors(
                region_idx, self.distance_matrix,
                radius_mm=self.spatial_radius,
                exclude_self=True
            )
            
            # Intersection: both same network AND spatial neighbors
            spatial_neighbor_mask = np.zeros(self.n_regions, dtype=bool)
            spatial_neighbor_mask[spatial_neighbors] = True
            
            intersection_mask = same_network_mask & spatial_neighbor_mask
            intersection_indices = np.where(intersection_mask)[0]
            
            if len(intersection_indices) > 0:
                diagonal_value = np.mean(m[region_idx, intersection_indices])
            else:
                # Fallback to hybrid if no intersection
                # Try network only
                same_network_indices = np.where(same_network_mask)[0]
                if len(same_network_indices) > 0:
                    diagonal_value = np.mean(m[region_idx, same_network_indices])
                else:
                    # Final fallback to row mean
                    row_sum = np.sum(m[region_idx, :]) - m[region_idx, region_idx]
                    diagonal_value = row_sum / (self.n_regions - 1)
            
            m[region_idx, region_idx] = diagonal_value
        
        return m
    
    # ============================================================
    # MAIN IMPUTATION METHOD
    # ============================================================
    
    def impute_diagonal(self, strategy: Optional[str] = None) -> np.ndarray:
        """
        Impute diagonal using specified strategy.
        
        Args:
            strategy: Strategy name (if None, uses config value)
            
        Returns:
            Matrix with imputed diagonal
        """
        chosen_strategy = strategy or self.strategy
        
        strategies = {
            'zero': self.replace_diagonal_with_zero,
            'one': self.replace_diagonal_with_one,
            'random': self.replace_diagonal_with_random,
            'mean': self.replace_diagonal_with_mean,
            'knn': self.replace_diagonal_with_knn,
            'network_mean': self.replace_diagonal_with_network_mean,
            'spatial_mean': self.replace_diagonal_with_spatial_mean,
            'hybrid': self.replace_diagonal_with_hybrid,
            'network_spatial_intersection': self.replace_diagonal_with_network_spatial_intersection
        }
        
        if chosen_strategy not in strategies:
            raise ValueError(
                f"Unknown strategy: {chosen_strategy}. "
                f"Available strategies: {list(strategies.keys())}"
            )
        
        print(f"Using diagonal imputation strategy: {chosen_strategy}")
        return strategies[chosen_strategy]()


# ============================================================
# PUBLIC API
# ============================================================

def impute_connectivity_diagonal(matrix: np.ndarray, 
                                region_list: Optional[List[str]] = None,
                                strategy: Optional[str] = None,
                                config_path: str = "config.yaml") -> np.ndarray:
    """
    Public API for diagonal imputation.
    
    Args:
        matrix: Connectivity matrix
        region_list: Optional list of region names (enables network/spatial strategies)
        strategy: Imputation strategy (overrides config)
        config_path: Path to configuration file
        
    Returns:
        Matrix with imputed diagonal
    """
    imputer = DiagonalImputer(matrix, region_list, config_path)
    return imputer.impute_diagonal(strategy)