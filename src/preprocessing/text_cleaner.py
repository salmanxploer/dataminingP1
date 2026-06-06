"""
Text Cleaning and Preprocessing
KDD PHASE 2: Data Cleaning

Handles all text preprocessing operations:
- Lowercasing
- Stopword removal
- Punctuation removal
- URL and email removal
- Noise removal

Theory:
-------
Text preprocessing is crucial for reducing noise and improving model performance.
Cleaning involves removing irrelevant information while preserving semantic meaning.
"""

import re
import logging
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    STOPWORDS_LANGUAGE,
    MIN_WORD_LENGTH,
    MAX_WORD_LENGTH,
    REMOVE_URLS,
    REMOVE_EMAILS,
    REMOVE_NUMBERS
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Text cleaning and normalization
    
    Objective: Remove noise and prepare text for analysis
    Algorithm: Multi-step cleaning pipeline
    """

    def __init__(self):
        """
        Initialize text cleaner with stopwords
        """
        self.stopwords = set(stopwords.words(STOPWORDS_LANGUAGE))
        logger.info(f"TextCleaner initialized with {len(self.stopwords)} stopwords")

    def clean_text(self, text: str) -> str:
        """
        Complete text cleaning pipeline
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Remove URLs
        if REMOVE_URLS:
            text = self._remove_urls(text)
        
        # Step 2: Remove emails
        if REMOVE_EMAILS:
            text = self._remove_emails(text)
        
        # Step 3: Convert to lowercase
        text = text.lower()
        
        # Step 4: Remove special characters and punctuation
        text = self._remove_special_chars(text)
        
        # Step 5: Remove numbers (optional)
        if REMOVE_NUMBERS:
            text = self._remove_numbers(text)
        
        # Step 6: Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    def _remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text
        
        Args:
            text: Input text
            
        Returns:
            Text without emails
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    def _remove_special_chars(self, text: str) -> str:
        """
        Remove special characters and punctuation
        
        Args:
            text: Input text
            
        Returns:
            Text with only alphanumeric characters and spaces
        """
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def _remove_numbers(self, text: str) -> str:
        """
        Remove numbers from text
        
        Args:
            text: Input text
            
        Returns:
            Text without numbers
        """
        return re.sub(r'\d+', '', text)

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text
            
        Returns:
            Text without stopwords
        """
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)

    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by length
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        return [
            token for token in tokens
            if MIN_WORD_LENGTH <= len(token) <= MAX_WORD_LENGTH
        ]

    def normalize_text(self, text: str) -> str:
        """
        Full normalization pipeline
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        return text


def clean_abstract(abstract: str) -> str:
    """
    Clean research paper abstract
    
    Args:
        abstract: Paper abstract text
        
    Returns:
        Cleaned abstract
    """
    cleaner = TextCleaner()
    return cleaner.clean_text(abstract)
