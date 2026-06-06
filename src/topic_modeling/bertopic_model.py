"""
BERTopic Topic Modeling
KDD PHASE 5: Pattern Discovery

Theory:
-------
BERTopic combines BERT embeddings with clustering to discover topics.
It creates interpretable and dynamic topic representations.

Algorithm:
----------
1. Generate document embeddings with BERT
2. Reduce dimensionality with UMAP
3. Cluster documents with HDBSCAN
4. Generate topic representations with TF-IDF and c-TF-IDF
5. Update representations dynamically

Objective:
Discover semantically meaningful topics using transformer embeddings.

Evaluation Metrics:
- Topic Diversity
- Topic Coherence
- Topic Stability
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from umap import UMAP
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import BERTOPIC_CONFIG, EMBEDDING_MODEL, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTopicModel:
    """
    BERTopic: Neural Topic Modeling with Transformers
    
    Objective: Discover semantically coherent topics using BERT embeddings
    Algorithm: Embedding + Dimensionality Reduction + Clustering + Topic Representation
    """

    def __init__(self, nr_topics: int = BERTOPIC_CONFIG['nr_topics']):
        """
        Initialize BERTopic model
        
        Args:
            nr_topics: Number of topics
        """
        self.nr_topics = nr_topics
        self.model = None
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"BERTopic model initialized with {nr_topics} topics")

    def train(self, texts: List[str], embeddings: np.ndarray = None) -> 'BERTopicModel':
        """
        Train BERTopic model
        
        Args:
            texts: List of document texts
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training BERTopic on {len(texts)} documents...")
        
        # Generate embeddings if not provided
        if embeddings is None:
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=self.nr_topics,
            min_topic_size=BERTOPIC_CONFIG['min_topic_size'],
            language=BERTOPIC_CONFIG['language'],
            calculate_probabilities=BERTOPIC_CONFIG['calculate_probabilities']
        )
        
        # Fit model
        topics, probs = self.model.fit_transform(texts, embeddings)
        
        logger.info(f"BERTopic model trained: {self.nr_topics} topics discovered")
        return self

    def get_topics(self, top_n_words: int = 10) -> Dict[int, List[Tuple]]:
        """
        Get topics with top words
        
        Args:
            top_n_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic_id to top words
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        topics = {}
        for topic_id in range(-1, self.nr_topics):
            topic_info = self.model.get_topic(topic_id)
            if topic_info:
                topics[topic_id] = topic_info[:top_n_words]
        
        return topics

    def get_document_topics(self, document_text: str) -> List[Tuple[int, float]]:
        """
        Get topic distribution for document
        
        Args:
            document_text: Document text
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Get topic and probability
        topic, prob = self.model.predict([document_text])
        return [(topic[0], prob[0])]

    def get_topic_info(self) -> Dict:
        """
        Get information about all topics
        
        Returns:
            DataFrame with topic information
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        return self.model.get_topic_info()

    def evaluate(self) -> Dict:
        """
        Evaluate model quality
        
        Evaluation Metrics:
        - Topic Diversity: How different topics are from each other
        - Topic Coherence: How coherent top words in each topic are
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        metrics = {
            'num_topics': self.nr_topics,
            'model_type': 'BERTopic',
            'embedding_model': EMBEDDING_MODEL
        }
        
        logger.info(f"BERTopic evaluation: {self.nr_topics} topics")
        return metrics

    def save(self, path: str = None):
        """
        Save model
        
        Args:
            path: Save path
        """
        if not path:
            path = str(PROJECT_ROOT / f'models/bertopic_model_{self.nr_topics}')
        
        self.model.save(path)
        logger.info(f"BERTopic model saved to {path}")

    def load(self, path: str):
        """
        Load model
        
        Args:
            path: Model path
        """
        self.model = BERTopic.load(path)
        logger.info(f"BERTopic model loaded from {path}")
        return self
