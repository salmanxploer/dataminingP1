"""
HDBSCAN Clustering
KDD PHASE 5: Pattern Discovery

Theory:
-------
HDBSCAN (Hierarchical Density-Based Spatial Clustering) is a density-based clustering algorithm.
It extends DBSCAN with hierarchical clustering to form a dendrogram of clusters.

Algorithm:
----------
1. Build a minimum spanning tree of the data
2. Compute core distances using k-distance graph
3. Construct dendrogram from MST
4. Extract flat clustering from dendrogram
5. Assign outliers to nearest clusters

Objective:
Discover natural clusters in data without specifying cluster count in advance.

Advantages:
- No need to specify number of clusters
- Can detect outliers (noise points)
- Discovers clusters of varying density and shape

Evaluation Metrics:
- Silhouette Score
- Davies-Bouldin Index
- Density-based cluster validity
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import HDBSCAN_CONFIG, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HDBSCANCluster:
    """
    HDBSCAN Hierarchical Density-Based Clustering
    
    Objective: Discover natural density-based clusters without pre-specifying count
    Algorithm: Hierarchical DBSCAN with dendrogram extraction
    """

    def __init__(self, min_cluster_size: int = HDBSCAN_CONFIG['min_cluster_size'],
                 min_samples: int = HDBSCAN_CONFIG['min_samples']):
        """
        Initialize HDBSCAN clusterer
        
        Args:
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood for core point
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = None
        self.labels = None
        self.probabilities = None
        logger.info(f"HDBSCAN initialized: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    def fit(self, features: np.ndarray) -> 'HDBSCANCluster':
        """
        Fit HDBSCAN model
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting HDBSCAN on {features.shape[0]} samples with {features.shape[1]} features...")
        
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=HDBSCAN_CONFIG['metric'],
            cluster_selection_method=HDBSCAN_CONFIG['cluster_selection_method']
        )
        
        self.labels = self.model.fit_predict(features)
        self.probabilities = self.model.probabilities_
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_outliers = list(self.labels).count(-1)
        
        logger.info(f"HDBSCAN completed: {n_clusters} clusters, {n_outliers} outliers")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels (-1 for outliers)
        """
        if not self.model:
            raise ValueError("Model not fitted yet")
        
        # Note: HDBSCAN doesn't have a standard predict method
        # Use approximate_predict for new samples
        labels, strengths = self.model.approximate_predict(features)
        return labels

    def evaluate(self, features: np.ndarray) -> Dict:
        """
        Evaluate clustering quality
        
        Evaluation Metrics:
        - Silhouette Score: Measure of cluster cohesion and separation
        - Davies-Bouldin Index: Average similarity ratio
        - Number of clusters: Automatically detected
        - Number of outliers: Points labeled as -1
        
        Args:
            features: Feature matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model or self.labels is None:
            raise ValueError("Model not fitted yet")
        
        # Filter out outliers for evaluation
        mask = self.labels != -1
        
        if mask.sum() == 0:
            logger.warning("No valid clusters found (all points are outliers)")
            return {'error': 'All points classified as outliers'}
        
        features_valid = features[mask]
        labels_valid = self.labels[mask]
        
        # Calculate metrics only on valid clusters
        silhouette = silhouette_score(features_valid, labels_valid)
        
        if len(set(labels_valid)) > 1:
            davies_bouldin = davies_bouldin_score(features_valid, labels_valid)
        else:
            davies_bouldin = 0.0
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_outliers = list(self.labels).count(-1)
        
        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_percentage': (n_outliers / len(self.labels)) * 100
        }
        
        logger.info(f"HDBSCAN evaluation: Silhouette={silhouette:.4f}, DB-Index={davies_bouldin:.4f}, "
                   f"Clusters={n_clusters}, Outliers={n_outliers}")
        return metrics

    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get cluster sizes (excluding outliers labeled as -1)
        
        Returns:
            Dictionary mapping cluster_id to size
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        
        unique, counts = np.unique(self.labels[self.labels != -1], return_counts=True)
        return dict(zip(unique, counts))

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """
        Get indices of documents in a cluster
        
        Args:
            cluster_id: Cluster ID (-1 for outliers)
            
        Returns:
            Array of document indices
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        
        return np.where(self.labels == cluster_id)[0]

    def get_outliers(self) -> np.ndarray:
        """
        Get indices of outlier points
        
        Returns:
            Array of outlier indices
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        
        return np.where(self.labels == -1)[0]

    def save(self, path: str = None):
        """
        Save model
        
        Args:
            path: Save path
        """
        import joblib
        
        if not path:
            path = str(PROJECT_ROOT / f'models/hdbscan_model.pkl')
        
        joblib.dump(self.model, path)
        logger.info(f"HDBSCAN model saved to {path}")

    def load(self, path: str):
        """
        Load model
        
        Args:
            path: Model path
        """
        import joblib
        
        self.model = joblib.load(path)
        logger.info(f"HDBSCAN model loaded from {path}")
        return self
