"""
Trend Analysis Module
KDD PHASE 6: Trend Analysis

Analyzes research trends over time:
- Topic frequency trends
- Emerging topics detection
- Trend strength calculation
- Temporal analysis
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyze research trends over time
    
    KDD Phase 6: Trend Analysis
    Objective: Identify emerging topics and trend patterns
    """

    def __init__(self):
        """
        Initialize trend analyzer
        """
        self.trends_data = {}
        logger.info("Trend analyzer initialized")

    def analyze_topic_trends(self, papers_by_date: Dict[str, List[int]], 
                            topic_assignments: Dict[int, int]) -> Dict:
        """
        Analyze topic frequency trends over time
        
        Args:
            papers_by_date: Dictionary mapping dates to paper indices
            topic_assignments: Dictionary mapping paper_id to topic_id
            
        Returns:
            Dictionary with trend analysis
        """
        trends = {}
        
        for date, paper_ids in papers_by_date.items():
            topic_counts = {}
            for paper_id in paper_ids:
                if paper_id in topic_assignments:
                    topic_id = topic_assignments[paper_id]
                    topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
            
            trends[date] = topic_counts
        
        self.trends_data = trends
        logger.info(f"Analyzed trends for {len(trends)} time periods")
        return trends

    def calculate_growth_rate(self, topic_id: int, window_size: int = 3) -> List[float]:
        """
        Calculate topic growth rate over time
        
        Args:
            topic_id: Topic ID to analyze
            window_size: Rolling window size for smoothing
            
        Returns:
            List of growth rates
        """
        # Extract topic frequencies over time
        dates = sorted(self.trends_data.keys())
        frequencies = [self.trends_data.get(date, {}).get(topic_id, 0) for date in dates]
        
        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(frequencies)):
            if frequencies[i-1] > 0:
                growth = (frequencies[i] - frequencies[i-1]) / frequencies[i-1]
            else:
                growth = 1.0 if frequencies[i] > 0 else 0.0
            growth_rates.append(growth)
        
        return growth_rates

    def detect_emerging_topics(self, threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        Detect emerging topics based on growth rate
        
        Emerging Topic Criteria:
        - Positive growth rate over recent periods
        - Emergence score above threshold
        
        Args:
            threshold: Emergence score threshold
            
        Returns:
            List of (topic_id, emergence_score) tuples
        """
        emerging = []
        
        # Get unique topics
        all_topics = set()
        for topic_counts in self.trends_data.values():
            all_topics.update(topic_counts.keys())
        
        # Calculate emergence score for each topic
        for topic_id in all_topics:
            growth_rates = self.calculate_growth_rate(topic_id)
            
            if len(growth_rates) > 0:
                # Emergence score = average recent growth rate
                recent_growth = np.mean(growth_rates[-3:]) if len(growth_rates) >= 3 else np.mean(growth_rates)
                
                if recent_growth > threshold:
                    emerging.append((topic_id, recent_growth))
        
        # Sort by emergence score
        emerging.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Detected {len(emerging)} emerging topics")
        
        return emerging

    def get_topic_timeline(self, topic_id: int) -> Dict[str, int]:
        """
        Get timeline for a specific topic
        
        Args:
            topic_id: Topic ID
            
        Returns:
            Dictionary mapping dates to frequencies
        """
        timeline = {}
        for date, topic_counts in self.trends_data.items():
            timeline[date] = topic_counts.get(topic_id, 0)
        
        return timeline

    def calculate_trend_strength(self, topic_id: int) -> float:
        """
        Calculate overall trend strength for a topic
        
        Metric: Standard deviation of normalized frequencies
        Higher values indicate stronger trends
        
        Args:
            topic_id: Topic ID
            
        Returns:
            Trend strength score (0-1)
        """
        timeline = self.get_topic_timeline(topic_id)
        frequencies = list(timeline.values())
        
        if len(frequencies) < 2 or sum(frequencies) == 0:
            return 0.0
        
        # Normalize frequencies
        max_freq = max(frequencies)
        normalized = [f / max_freq for f in frequencies]
        
        # Calculate coefficient of variation
        strength = np.std(normalized) / (np.mean(normalized) + 1e-10)
        return min(strength, 1.0)

    def get_trend_report(self) -> str:
        """
        Generate trend analysis report
        
        Returns:
            Formatted trend report
        """
        report = "=" * 60 + "\n"
        report += "TREND ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Analysis Period: {len(self.trends_data)} time periods\n"
        
        # Get emerging topics
        emerging = self.detect_emerging_topics()
        report += f"\nTop Emerging Topics:\n"
        for i, (topic_id, score) in enumerate(emerging[:5], 1):
            report += f"  {i}. Topic {topic_id}: Emergence Score = {score:.4f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        return report
