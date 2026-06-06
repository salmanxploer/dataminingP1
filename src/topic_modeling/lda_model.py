"""
Latent Dirichlet Allocation (LDA) Topic Modeling
KDD PHASE 5: Pattern Discovery

Theory:
-------
LDA is a generative probabilistic model for collections of discrete data.
It assumes each document is a mixture of topics, and each topic is a mixture of words.

Algorithm:
----------
1. Initialize: Randomly assign topics to words
2. Iterate: Update topic-word and document-topic distributions
3. Convergence: Check for stable distributions

Objective:
Discover latent topics in research papers to understand research trends.

Evaluation Metrics:
- Coherence Score (higher is better, range: -1 to 1)
- Perplexity (lower is better)
"""

import logging
from typing import List, Tuple, Dict
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import LDA_CONFIG, COHERENCE_METRIC, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LDATopicModel:
    """
    Latent Dirichlet Allocation Topic Modeling
    
    Objective: Discover hidden topics in document collections
    Algorithm: Variational Bayes with Gibbs Sampling
    """

    def __init__(self, num_topics: int = LDA_CONFIG['num_topics']):
        """
        Initialize LDA model
        
        Args:
            num_topics: Number of topics to discover
        """
        self.num_topics = num_topics
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.coherence_score = None
        logger.info(f"LDA model initialized with {num_topics} topics")

    def prepare_corpus(self, texts: List[str]) -> Tuple:
        """
        Prepare corpus for LDA
        
        Args:
            texts: List of document texts
            
        Returns:
            Tuple of (dictionary, corpus)
        """
        # Tokenize
        tokenized_texts = [text.split() for text in texts]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_texts)
        logger.info(f"Dictionary created with {len(self.dictionary)} tokens")
        
        # Create corpus (bag of words)
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        logger.info(f"Corpus created with {len(self.corpus)} documents")
        
        return self.dictionary, self.corpus

    def train(self, texts: List[str]) -> 'LDATopicModel':
        """
        Train LDA model
        
        Args:
            texts: List of document texts
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training LDA model on {len(texts)} documents...")
        
        # Prepare corpus
        self.prepare_corpus(texts)
        
        # Train model
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=LDA_CONFIG['passes'],
            workers=LDA_CONFIG['workers'],
            per_word_topics=LDA_CONFIG['per_word_topics'],
            minimum_probability=LDA_CONFIG['minimum_probability']
        )
        
        logger.info("LDA model trained successfully")
        return self

    def evaluate(self, texts: List[str]) -> Dict:
        """
        Evaluate model using coherence score
        
        Evaluation Metric: Coherence
        - Measures semantic consistency of top words in each topic
        - Range: -1 to 1 (higher is better)
        - Typical good coherence: > 0.5
        
        Args:
            texts: List of document texts
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=self.model,
            texts=[text.split() for text in texts],
            dictionary=self.dictionary,
            coherence=COHERENCE_METRIC
        )
        self.coherence_score = coherence_model.get_coherence()
        
        # Calculate perplexity
        perplexity = self.model.log_perplexity(self.corpus)
        
        metrics = {
            'coherence_score': self.coherence_score,
            'perplexity': perplexity,
            'num_topics': self.num_topics,
            'vocabulary_size': len(self.dictionary)
        }
        
        logger.info(f"Model evaluation: Coherence={self.coherence_score:.4f}, Perplexity={perplexity:.4f}")
        return metrics

    def get_topics(self, num_words: int = 10) -> List[Tuple]:
        """
        Get top words for each topic
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List of topics with top words
        """
        if not self.model:
            raise ValueError("Model not trained yet")
        
        topics = []
        for topic_id in range(self.num_topics):
            top_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append((topic_id, top_words))
        
        return topics

    def get_document_topics(self, document_text: str) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document
        
        Args:
            document_text: Document text
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if not self.model or not self.dictionary:
            raise ValueError("Model not trained yet")
        
        # Convert to bag of words
        bow = self.dictionary.doc2bow(document_text.split())
        
        # Get topic distribution
        topics = self.model.get_document_topics(bow)
        return sorted(topics, key=lambda x: x[1], reverse=True)

    def save(self, path: str = None):
        """
        Save model to disk
        
        Args:
            path: Save path
        """
        if not path:
            path = str(PROJECT_ROOT / f'models/lda_model_{self.num_topics}.model')
        
        self.model.save(path)
        self.dictionary.save(path.replace('.model', '.dictionary'))
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Model path
        """
        self.model = LdaModel.load(path)
        self.dictionary = corpora.Dictionary.load(path.replace('.model', '.dictionary'))
        logger.info(f"Model loaded from {path}")
        return self
