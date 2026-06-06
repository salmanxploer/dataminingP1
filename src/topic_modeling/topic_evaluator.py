"""
Topic Model Evaluator
KDD PHASE 5: Pattern Discovery

Evaluation metrics for topic models:
- Coherence Score
- Perplexity
- Topic Diversity
- Topic Stability
"""

import logging
from typing import Dict, List
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicEvaluator:
    """
    Evaluate topic model quality
    
    Evaluation Metrics:
    - Coherence: Semantic consistency of top words (-1 to 1, higher better)
    - Perplexity: Model fit to data (lower better)
    - Topic Diversity: Uniqueness of topics
    """

    @staticmethod
    def evaluate_coherence(coherence_score: float) -> Dict:
        """
        Interpret coherence score
        
        Args:
            coherence_score: Coherence score from model
            
        Returns:
            Evaluation dictionary
        """
        if coherence_score < 0.3:
            quality = "Poor"
        elif coherence_score < 0.5:
            quality = "Fair"
        elif coherence_score < 0.7:
            quality = "Good"
        else:
            quality = "Excellent"
        
        return {
            'coherence_score': coherence_score,
            'quality': quality,
            'recommendation': f"Coherence score is {quality.lower()}. Consider adjusting num_topics."
        }

    @staticmethod
    def compare_models(models: Dict[str, Dict]) -> Dict:
        """
        Compare multiple topic models
        
        Args:
            models: Dictionary of model evaluations
            
        Returns:
            Comparison results
        """
        comparison = {
            'best_model': None,
            'best_coherence': -1,
            'models': {}
        }
        
        for model_name, metrics in models.items():
            comparison['models'][model_name] = metrics
            if metrics.get('coherence_score', -1) > comparison['best_coherence']:
                comparison['best_coherence'] = metrics['coherence_score']
                comparison['best_model'] = model_name
        
        return comparison
