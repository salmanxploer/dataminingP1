"""
arXiv Data Collector
KDD PHASE 1: Data Collection

Collects AI research papers from arXiv API.
Focuses on papers from the last 12 months.
"""

import requests
import logging
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import feedparser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    ARXIV_BASE_URL,
    ARXIV_SEARCH_QUERY,
    MAX_PAPERS,
    PAPERS_PER_REQUEST,
    RAW_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArXivCollector:
    """
    Collects research papers from arXiv API
    
    Theory:
    -------
    arXiv is an open repository of electronic preprints and postprints.
    The API uses atom feeds and supports complex queries.
    
    Objective:
    ----------
    Extract paper metadata including title, authors, abstract, and dates.
    
    KDD Phase: DATA COLLECTION
    """

    def __init__(self, search_query: str = ARXIV_SEARCH_QUERY):
        """
        Initialize arXiv collector
        
        Args:
            search_query: arXiv search query string
        """
        self.base_url = ARXIV_BASE_URL
        self.search_query = search_query
        self.papers = []
        self.total_results = 0

    def search(self, max_results: int = MAX_PAPERS) -> List[Dict]:
        """
        Search arXiv for papers
        
        Args:
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of paper dictionaries with metadata
        """
        logger.info(f"Starting arXiv search for: {self.search_query}")
        
        try:
            # Query parameters
            params = {
                'search_query': self.search_query,
                'start': 0,
                'max_results': min(PAPERS_PER_REQUEST, max_results),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            papers_collected = 0
            start_index = 0
            
            while papers_collected < max_results:
                params['start'] = start_index
                params['max_results'] = min(PAPERS_PER_REQUEST, max_results - papers_collected)
                
                logger.info(f"Fetching papers {start_index} to {start_index + params['max_results']}...")
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse feed
                feed = feedparser.parse(response.content)
                
                if not feed.entries:
                    logger.info("No more papers found")
                    break
                
                # Extract papers
                for entry in feed.entries:
                    paper = self._parse_entry(entry)
                    self.papers.append(paper)
                    papers_collected += 1
                
                start_index += params['max_results']
                
                # Check total results
                if 'opensearch_totalresults' in feed.feed:
                    self.total_results = int(feed.feed.opensearch_totalresults)
                    logger.info(f"Total available papers: {self.total_results}")
            
            logger.info(f"Successfully collected {papers_collected} papers from arXiv")
            return self.papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from arXiv: {str(e)}")
            raise

    def _parse_entry(self, entry) -> Dict:
        """
        Parse arXiv feed entry
        
        Args:
            entry: Feed entry object
            
        Returns:
            Dictionary with paper metadata
        """
        paper = {
            'arxiv_id': entry.id.split('/abs/')[-1],
            'title': entry.title,
            'abstract': entry.summary,
            'authors': [author.name for author in entry.authors],
            'publication_date': entry.published,
            'url': entry.id,
            'source': 'arxiv',
            'categories': entry.arxiv_primary_category.get('term', 'Unknown') if hasattr(entry, 'arxiv_primary_category') else 'Unknown'
        }
        return paper

    def save_to_file(self, filename: str = "arxiv_papers.json") -> Path:
        """
        Save collected papers to JSON file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = RAW_DATA_DIR / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.papers, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved {len(self.papers)} papers to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving papers: {str(e)}")
            raise


def main():
    """Main execution function"""
    collector = ArXivCollector()
    papers = collector.search(max_results=100)  # Start with 100 for testing
    collector.save_to_file()
    
    logger.info(f"Collected {len(papers)} papers")
    if papers:
        logger.info(f"Sample paper: {papers[0]['title']}")


if __name__ == "__main__":
    main()
