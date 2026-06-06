"""
Database Manager for Data Collection
KDD PHASE 1 & 3: Data Collection and Integration

Handles storage of collected papers in the database.
"""

import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from database.init_db import DatabaseInitializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperDatabaseManager:
    """
    Manages paper data storage in database
    
    KDD Phase 1: DATA COLLECTION - Raw data storage
    KDD Phase 3: DATA INTEGRATION - Consolidating data from multiple sources
    """

    def __init__(self, db_initializer: Optional[DatabaseInitializer] = None):
        """
        Initialize database manager
        
        Args:
            db_initializer: DatabaseInitializer instance
        """
        self.db = db_initializer or DatabaseInitializer()
        self.db.create_engine()

    def insert_papers(self, papers: List[Dict], source: str = 'arxiv') -> int:
        """
        Insert papers into database
        
        Args:
            papers: List of paper dictionaries
            source: Data source (arxiv, semantic_scholar)
            
        Returns:
            Number of papers inserted
        """
        session = self.db.get_session()
        inserted_count = 0
        
        try:
            for paper in papers:
                # Check if paper already exists
                existing = session.execute(
                    text("SELECT paper_id FROM papers WHERE url = :url"),
                    {"url": paper.get('url', '')}
                ).fetchone()
                
                if existing:
                    logger.info(f"Paper {paper.get('title', 'Unknown')} already exists")
                    continue
                
                # Insert paper
                session.execute(
                    text("""
                        INSERT INTO papers (
                            arxiv_id, title, abstract, authors, 
                            publication_date, source, url
                        ) VALUES (
                            :arxiv_id, :title, :abstract, :authors,
                            :publication_date, :source, :url
                        )
                    """),
                    {
                        'arxiv_id': paper.get('arxiv_id'),
                        'title': paper.get('title'),
                        'abstract': paper.get('abstract'),
                        'authors': json.dumps(paper.get('authors', [])),
                        'publication_date': paper.get('publication_date'),
                        'source': source,
                        'url': paper.get('url')
                    }
                )
                inserted_count += 1
            
            session.commit()
            logger.info(f"Inserted {inserted_count} papers into database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting papers: {str(e)}")
            raise
        finally:
            session.close()
        
        return inserted_count

    def get_paper_count(self) -> int:
        """
        Get total number of papers in database
        
        Returns:
            Total paper count
        """
        session = self.db.get_session()
        try:
            result = session.execute(text("SELECT COUNT(*) FROM papers")).fetchone()
            return result[0] if result else 0
        finally:
            session.close()

    def get_papers_by_source(self, source: str) -> List[Dict]:
        """
        Retrieve papers by source
        
        Args:
            source: Data source name
            
        Returns:
            List of paper dictionaries
        """
        session = self.db.get_session()
        try:
            results = session.execute(
                text("SELECT * FROM papers WHERE source = :source"),
                {"source": source}
            ).fetchall()
            return [dict(result._mapping) for result in results]
        finally:
            session.close()

    def log_processing(self, operation: str, status: str, papers_processed: int, 
                      execution_time: float, error_msg: Optional[str] = None):
        """
        Log data processing operation
        
        Args:
            operation: Operation name
            status: SUCCESS or FAILED
            papers_processed: Number of papers processed
            execution_time: Execution time in seconds
            error_msg: Error message if any
        """
        session = self.db.get_session()
        try:
            session.execute(
                text("""
                    INSERT INTO processing_logs (
                        operation, status, papers_processed, 
                        execution_time_seconds, error_message
                    ) VALUES (
                        :operation, :status, :papers_processed,
                        :execution_time, :error_message
                    )
                """),
                {
                    'operation': operation,
                    'status': status,
                    'papers_processed': papers_processed,
                    'execution_time': execution_time,
                    'error_message': error_msg
                }
            )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging processing: {str(e)}")
        finally:
            session.close()
