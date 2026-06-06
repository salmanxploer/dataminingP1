"""
Visualization Module
KDD PHASE 7: Visualization & Reporting

Comprehensive visualization for KDD analysis results:
- Topic distribution plots
- Trend visualizations
- Cluster visualizations
- Heatmaps and networks
- Interactive dashboards
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, VISUALIZATION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figure_size']
plt.rcParams['font.size'] = VISUALIZATION_CONFIG['font_size']


class Visualizer:
    """
    Create visualizations for KDD analysis results
    
    Visualization Types:
    - Topic distributions
    - Trend curves
    - Cluster scatter plots
    - Heatmaps
    - Bar charts
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / 'visualizations')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer initialized. Output: {self.output_dir}")

    def plot_topic_distribution(self, topic_counts: Dict[int, int], 
                               title: str = "Topic Distribution") -> str:
        """
        Plot topic frequency distribution
        
        Args:
            topic_counts: Dictionary mapping topic_id to count
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        topics = sorted(topic_counts.keys())
        counts = [topic_counts[t] for t in topics]
        
        ax.bar([str(t) for t in topics], counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"topic_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Topic distribution plot saved: {output_path}")
        return str(output_path)

    def plot_trend_curve(self, timeline: Dict[str, int], topic_id: int = None,
                        title: str = "Trend Over Time") -> str:
        """
        Plot trend curve over time
        
        Args:
            timeline: Dictionary mapping dates to frequencies
            topic_id: Topic ID (for title)
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        dates = sorted(timeline.keys())
        values = [timeline[d] for d in dates]
        
        # Convert date strings to datetime
        date_objs = pd.to_datetime(dates)
        
        ax.plot(date_objs, values, marker='o', linestyle='-', color='darkgreen', 
               linewidth=2, markersize=6, label='Frequency')
        
        # Add trend line
        z = np.polyfit(range(len(values)), values, 2)
        p = np.poly1d(z)
        ax.plot(date_objs, p(range(len(values))), "--", color='red', 
               linewidth=2, alpha=0.7, label='Trend')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / f"trend_topic{topic_id or 'all'}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trend plot saved: {output_path}")
        return str(output_path)

    def plot_cluster_scatter(self, features: np.ndarray, labels: np.ndarray,
                            title: str = "Cluster Visualization") -> str:
        """
        Plot 2D scatter plot of clusters
        
        Note: Uses first two features for visualization
        For high-dimensional data, consider using PCA/t-SNE first
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        # If features have more than 2 dimensions, use first 2 (or could apply PCA)
        if features.shape[1] > 2:
            logger.warning("Features have >2 dimensions. Using first 2 for visualization.")
            features_2d = features[:, :2]
        else:
            features_2d = features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot clusters
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(sorted(unique_labels), colors):
            if label == -1:
                # Outliers
                color = 'black'
                marker = 'x'
                label_str = 'Outliers'
            else:
                marker = 'o'
                label_str = f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[color], label=label_str, marker=marker, s=50, alpha=0.7)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "cluster_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster scatter plot saved: {output_path}")
        return str(output_path)

    def plot_cluster_sizes(self, cluster_sizes: Dict[int, int],
                          title: str = "Cluster Sizes") -> str:
        """
        Plot cluster size distribution
        
        Args:
            cluster_sizes: Dictionary mapping cluster_id to size
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        clusters = sorted(cluster_sizes.keys())
        sizes = [cluster_sizes[c] for c in clusters]
        
        ax.bar([str(c) for c in clusters], sizes, color='coral', alpha=0.7)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Documents')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "cluster_sizes.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster sizes plot saved: {output_path}")
        return str(output_path)

    def plot_forecast_comparison(self, actual: np.ndarray, forecast: np.ndarray,
                                dates: List[str], title: str = "Forecast Comparison") -> str:
        """
        Plot actual vs forecasted values
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            dates: Date labels
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        date_objs = pd.to_datetime(dates)
        
        ax.plot(date_objs[:len(actual)], actual, marker='o', label='Actual',
               color='blue', linewidth=2, markersize=6)
        ax.plot(date_objs[len(actual):len(actual)+len(forecast)], forecast,
               marker='s', label='Forecast', color='red', linewidth=2, 
               markersize=6, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / "forecast_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forecast comparison plot saved: {output_path}")
        return str(output_path)

    def plot_heatmap(self, data: np.ndarray, 
                    x_labels: List[str] = None,
                    y_labels: List[str] = None,
                    title: str = "Heatmap") -> str:
        """
        Plot heatmap
        
        Args:
            data: 2D data array
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels,
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Value'})
        
        ax.set_title(title)
        plt.tight_layout()
        
        output_path = self.output_dir / "heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved: {output_path}")
        return str(output_path)

    def create_summary_dashboard(self, metrics_dict: Dict) -> str:
        """
        Create summary dashboard with multiple metrics
        
        Args:
            metrics_dict: Dictionary of metric names and values
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Add text metrics
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off')
        
        text_str = "KDD Analysis Summary\n" + "="*50 + "\n"
        for key, value in list(metrics_dict.items())[:5]:
            if isinstance(value, float):
                text_str += f"{key}: {value:.4f}\n"
            else:
                text_str += f"{key}: {value}\n"
        
        ax_text.text(0.1, 0.5, text_str, fontsize=12, family='monospace',
                    verticalalignment='center')
        
        # Placeholder for additional metrics visualization
        for i in range(1, 6):
            ax = fig.add_subplot(gs[i // 2 + 1, i % 2])
            ax.text(0.5, 0.5, f'Metric {i}', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('KDD Analysis Dashboard', fontsize=16, fontweight='bold')
        
        output_path = self.output_dir / "summary_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved: {output_path}")
        return str(output_path)
