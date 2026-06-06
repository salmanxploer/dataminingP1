"""
Database Initialization Module
KDD PHASE 3: Data Integration

Initializes SQLite/PostgreSQL database with schema and sample data.
Handles connection management and migration.
"""

import sqlite3
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATABASE_URL, PROJECT_ROOT
from database.schema import CREATE_TABLES_SQL, INSERT_SAMPLE_DATA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Manages database initialization and migrations"""

    def __init__(self, db_url: str = DATABASE_URL):
        """
        Initialize database manager
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.engine = None
        self.SessionLocal = None

    def create_engine(self):
        """Create SQLAlchemy engine"""
        try:
            self.engine = create_engine(self.db_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info(f"Database engine created: {self.db_url}")
            return self.engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {str(e)}")
            raise

    def initialize_schema(self):
        """Create all tables from schema"""
        if not self.engine:
            self.create_engine()
        
        try:
            with self.engine.connect() as connection:
                # Execute schema creation
                for statement in CREATE_TABLES_SQL.split(';'):
                    if statement.strip():
                        connection.execute(text(statement))
                connection.commit()
            logger.info("Database schema created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {str(e)}")
            raise

    def insert_sample_data(self):
        """Insert initial sample data"""
        if not self.engine:
            self.create_engine()
        
        try:
            with self.engine.connect() as connection:
                for statement in INSERT_SAMPLE_DATA.split(';'):
                    if statement.strip():
                        connection.execute(text(statement))
                connection.commit()
            logger.info("Sample data inserted successfully")
        except Exception as e:
            logger.error(f"Failed to insert sample data: {str(e)}")
            raise

    def get_session(self) -> Session:
        """Get database session"""
        if not self.SessionLocal:
            self.create_engine()
        return self.SessionLocal()

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def reset_database(self):
        """Drop all tables and recreate (WARNING: Data loss)"""
        if not self.engine:
            self.create_engine()
        
        try:
            # Drop all tables
            from sqlalchemy import MetaData
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
            
            # Recreate schema
            self.initialize_schema()
            logger.info("Database reset completed")
        except Exception as e:
            logger.error(f"Failed to reset database: {str(e)}")
            raise


def main():
    """Initialize database"""
    logger.info("Starting database initialization...")
    
    initializer = DatabaseInitializer()
    initializer.create_engine()
    initializer.initialize_schema()
    initializer.insert_sample_data()
    
    logger.info("Database initialization completed successfully")


if __name__ == "__main__":
    main()
