"""
Cluster Evaluator
KDD PHASE 5: Pattern Discovery

Comprehensive evaluation metrics for clustering results:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Cluster stability
- Cluster quality metrics
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Evaluate clustering quality and stability
    
    Evaluation Metrics:
    - Silhouette Score: Cohesion and separation (-1 to 1, higher better)
    - Davies-Bouldin Index: Cluster separation (lower better)
    - Calinski-Harabasz: Between/within cluster dispersion (higher better)
    """

    @staticmethod
    def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Comprehensive clustering evaluation
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster labels
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Filter out noise points (label == -1)
        mask = labels != -1
        
        if mask.sum() == 0:
            logger.warning("No valid clusters found")
            return {'error': 'No valid clusters'}
        
        features_valid = features[mask]
        labels_valid = labels[mask]
        
        metrics = {}
        
        # Silhouette Score
        try:
            silhouette = silhouette_score(features_valid, labels_valid)
            metrics['silhouette_score'] = silhouette
            metrics['silhouette_quality'] = ClusterEvaluator._interpret_silhouette(silhouette)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {str(e)}")
            metrics['silhouette_score'] = None
        
        # Davies-Bouldin Index
        try:
            if len(set(labels_valid)) > 1:
                davies_bouldin = davies_bouldin_score(features_valid, labels_valid)
                metrics['davies_bouldin_score'] = davies_bouldin
                metrics['davies_bouldin_quality'] = ClusterEvaluator._interpret_davies_bouldin(davies_bouldin)
            else:
                metrics['davies_bouldin_score'] = 0
                metrics['davies_bouldin_quality'] = 'Single cluster'
        except Exception as e:
            logger.error(f"Error calculating Davies-Bouldin: {str(e)}")
            metrics['davies_bouldin_score'] = None
        
        # Calinski-Harabasz Score
        try:
            calinski_harabasz = calinski_harabasz_score(features_valid, labels_valid)
            metrics['calinski_harabasz_score'] = calinski_harabasz
            metrics['calinski_harabasz_quality'] = ClusterEvaluator._interpret_calinski_harabasz(calinski_harabasz)
        except Exception as e:
            logger.error(f"Error calculating Calinski-Harabasz: {str(e)}")
            metrics['calinski_harabasz_score'] = None
        
        # Cluster distribution
        unique, counts = np.unique(labels_valid, return_counts=True)
        metrics['num_clusters'] = len(unique)
        metrics['cluster_sizes'] = dict(zip(unique, counts))
        metrics['cluster_distribution'] = {
            'min_size': int(counts.min()),
            'max_size': int(counts.max()),
            'mean_size': float(counts.mean()),
            'std_size': float(counts.std())
        }
        
        # Noise points (if applicable)
        n_outliers = (labels == -1).sum()
        if n_outliers > 0:
            metrics['n_outliers'] = n_outliers
            metrics['outlier_percentage'] = (n_outliers / len(labels)) * 100
        
        logger.info(f"Clustering evaluation: Silhouette={metrics.get('silhouette_score'):.4f}, "
                   f"Davies-Bouldin={metrics.get('davies_bouldin_score'):.4f}")
        
        return metrics

    @staticmethod
    def _interpret_silhouette(score: float) -> str:
        """
        Interpret silhouette score
        
        Args:
            score: Silhouette score (-1 to 1)
            
        Returns:
            Quality interpretation
        """
        if score < -0.1:
            return "Very Poor"
        elif score < 0.2:
            return "Poor"
        elif score < 0.5:
            return "Fair"
        elif score < 0.7:
            return "Good"
        else:
            return "Excellent"

    @staticmethod
    def _interpret_davies_bouldin(score: float) -> str:
        """
        Interpret Davies-Bouldin index
        
        Args:
            score: Davies-Bouldin score (lower is better)
            
        Returns:
            Quality interpretation
        """
        if score > 2.0:
            return "Poor"
        elif score > 1.5:
            return "Fair"
        elif score > 1.0:
            return "Good"
        else:
            return "Excellent"

    @staticmethod
    def _interpret_calinski_harabasz(score: float) -> str:
        """
        Interpret Calinski-Harabasz score
        
        Args:
            score: Calinski-Harabasz score (higher is better)
            
        Returns:
            Quality interpretation
        """
        if score < 10:
            return "Poor"
        elif score < 50:
            return "Fair"
        elif score < 100:
            return "Good"
        else:
            return "Excellent"

    @staticmethod
    def compare_clusterings(features: np.ndarray, labels1: np.ndarray, 
                           labels2: np.ndarray) -> Dict:
        """
        Compare two clustering results
        
        Args:
            features: Feature matrix
            labels1: First clustering labels
            labels2: Second clustering labels
            
        Returns:
            Comparison metrics
        """
        # Remove noise points for comparison
        mask = (labels1 != -1) & (labels2 != -1)
        labels1_valid = labels1[mask]
        labels2_valid = labels2[mask]
        
        comparison = {
            'adjusted_rand_score': adjusted_rand_score(labels1_valid, labels2_valid),
            'normalized_mutual_info': normalized_mutual_info_score(labels1_valid, labels2_valid)
        }
        
        # Evaluate both separately
        comparison['clustering1_metrics'] = ClusterEvaluator.evaluate_clustering(features, labels1)
        comparison['clustering2_metrics'] = ClusterEvaluator.evaluate_clustering(features, labels2)
        
        return comparison

    @staticmethod
    def get_cluster_quality_report(features: np.ndarray, labels: np.ndarray) -> str:
        """
        Generate detailed cluster quality report
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            
        Returns:
            Formatted quality report
        """
        metrics = ClusterEvaluator.evaluate_clustering(features, labels)
        
        report = "=" * 60 + "\n"
        report += "CLUSTERING QUALITY REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Number of Clusters: {metrics.get('num_clusters', 'N/A')}\n"
        report += f"Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f} ({metrics.get('silhouette_quality', 'N/A')})\n"
        report += f"Davies-Bouldin Index: {metrics.get('davies_bouldin_score', 'N/A'):.4f} ({metrics.get('davies_bouldin_quality', 'N/A')})\n"
        report += f"Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.2f} ({metrics.get('calinski_harabasz_quality', 'N/A')})\n"
        
        if 'cluster_distribution' in metrics:
            dist = metrics['cluster_distribution']
            report += f"\nCluster Size Distribution:\n"
            report += f"  Min: {dist['min_size']}, Max: {dist['max_size']}, Mean: {dist['mean_size']:.2f}\n"
        
        if 'outlier_percentage' in metrics:
            report += f"\nOutliers: {metrics['n_outliers']} ({metrics['outlier_percentage']:.2f}%)\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
