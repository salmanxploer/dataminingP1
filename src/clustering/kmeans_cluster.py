"""
K-Means Clustering
KDD PHASE 5: Pattern Discovery

Theory:
-------
K-Means is a centroid-based clustering algorithm that partitions data into k clusters.
Each data point belongs to the cluster with the nearest mean.

Algorithm:
----------
1. Initialize: Randomly select k cluster centers
2. Assign: Assign each point to the nearest center
3. Update: Recalculate cluster centers based on assigned points
4. Repeat: Until convergence

Objective:
Group similar research papers into k clusters for document organization.

Evaluation Metrics:
- Silhouette Score (range: -1 to 1, higher is better)
- Davies-Bouldin Index (lower is better)
- Within-cluster sum of squares (WCSS)
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import KMEANS_CONFIG, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KMeansCluster:
    """
    K-Means Document Clustering
    
    Objective: Partition documents into k semantically coherent clusters
    Algorithm: Lloyd's algorithm (k-means++)
    """

    def __init__(self, n_clusters: int = KMEANS_CONFIG['n_clusters']):
        """
        Initialize K-Means clusterer
        
        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        self.model = None
        self.labels = None
        self.centers = None
        logger.info(f"K-Means initialized with {n_clusters} clusters")

    def fit(self, features: np.ndarray) -> 'KMeansCluster':
        """
        Fit K-Means model
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting K-Means on {features.shape[0]} samples with {features.shape[1]} features...")
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=KMEANS_CONFIG['init'],
            n_init=KMEANS_CONFIG['n_init'],
            max_iter=KMEANS_CONFIG['max_iter'],
            random_state=KMEANS_CONFIG['random_state']
        )
        
        self.labels = self.model.fit_predict(features)
        self.centers = self.model.cluster_centers_
        
        logger.info(f"K-Means fitting completed. WCSS: {self.model.inertia_:.4f}")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels
        """
        if not self.model:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(features)

    def evaluate(self, features: np.ndarray) -> Dict:
        """
        Evaluate clustering quality
        
        Evaluation Metrics:
        - Silhouette Score: Measures how similar points are to their cluster vs other clusters
          Range: -1 to 1 (higher is better)
        - Davies-Bouldin Index: Average similarity between each cluster and its most similar cluster
          (lower is better)
        - Calinski-Harabasz Score: Ratio of between-cluster to within-cluster dispersion
          (higher is better)
        
        Args:
            features: Feature matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model or self.labels is None:
            raise ValueError("Model not fitted yet")
        
        silhouette = silhouette_score(features, self.labels)
        davies_bouldin = davies_bouldin_score(features, self.labels)
        calinski_harabasz = calinski_harabasz_score(features, self.labels)
        
        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': self.n_clusters,
            'wcss': self.model.inertia_
        }
        
        logger.info(f"K-Means evaluation: Silhouette={silhouette:.4f}, Davies-Bouldin={davies_bouldin:.4f}")
        return metrics

    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get cluster sizes
        
        Returns:
            Dictionary mapping cluster_id to size
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """
        Get indices of documents in a cluster
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Array of document indices
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet")
        
        return np.where(self.labels == cluster_id)[0]

    def save(self, path: str = None):
        """
        Save model
        
        Args:
            path: Save path
        """
        import joblib
        
        if not path:
            path = str(PROJECT_ROOT / f'models/kmeans_model_{self.n_clusters}.pkl')
        
        joblib.dump(self.model, path)
        logger.info(f"K-Means model saved to {path}")

    def load(self, path: str):
        """
        Load model
        
        Args:
            path: Model path
        """
        import joblib
        
        self.model = joblib.load(path)
        logger.info(f"K-Means model loaded from {path}")
        return self
