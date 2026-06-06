"""
Database Schema Definition
Entity-Relationship Model for AI Research Trend Mining System

KDD PHASE 3: DATA INTEGRATION
This module handles the consolidation of data from multiple sources
into a unified relational database.
"""

CREATE_TABLES_SQL = """

-- Papers Table: Core collection of research papers
-- KDD PHASE 3: Stores integrated data from multiple sources
CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id VARCHAR(50) UNIQUE,
    semantic_scholar_id VARCHAR(100) UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,  -- JSON array of author names
    publication_date DATE,
    source VARCHAR(50),  -- 'arxiv' or 'semantic_scholar'
    url TEXT UNIQUE,
    pdf_url TEXT,
    citation_count INTEGER DEFAULT 0,
    influence_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Research Categories Table
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parent_category_id INTEGER REFERENCES categories(category_id)
);

-- Paper-Category Junction Table (Many-to-Many)
CREATE TABLE IF NOT EXISTS paper_categories (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(category_id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, category_id)
);

-- KDD PHASE 5: PATTERN DISCOVERY
-- Topics Table: Topics discovered via LDA and BERTopic
CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_label VARCHAR(100),
    top_keywords TEXT,  -- JSON array of keywords
    topic_description TEXT,
    algorithm VARCHAR(50),  -- 'LDA' or 'BERTopic'
    coherence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper-Topic Junction Table (Many-to-Many)
-- Stores topic assignments with probabilities
CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    probability FLOAT NOT NULL,  -- Topic probability (0-1)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (paper_id, topic_id)
);

-- KDD PHASE 5: PATTERN DISCOVERY
-- Clusters Table: Document clusters from K-Means and HDBSCAN
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_label INTEGER,
    cluster_name VARCHAR(100),
    size INTEGER,
    algorithm VARCHAR(50),  -- 'KMeans' or 'HDBSCAN'
    silhouette_score FLOAT,
    davies_bouldin_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper-Cluster Junction Table (Many-to-Many)
CREATE TABLE IF NOT EXISTS paper_clusters (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    distance_to_center FLOAT,
    PRIMARY KEY (paper_id, cluster_id)
);

-- KDD PHASE 6: KNOWLEDGE DISCOVERY
-- Trends Table: Topic frequency and growth analysis over time
CREATE TABLE IF NOT EXISTS trends (
    trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    year_month VARCHAR(7),  -- Format: YYYY-MM
    frequency INTEGER,
    growth_rate FLOAT,
    emerging_score FLOAT,  -- Emerging topic detection score
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- KDD PHASE 7: PREDICTIVE DATA MINING
-- Forecasts Table: Future trend predictions
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    forecast_date DATE,
    predicted_frequency FLOAT,
    lower_bound FLOAT,
    upper_bound FLOAT,
    confidence_interval FLOAT,
    forecast_model VARCHAR(50),  -- 'Prophet' or 'ARIMA'
    mape FLOAT,  -- Mean Absolute Percentage Error
    rmse FLOAT,  -- Root Mean Squared Error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Author Analysis Table
CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) UNIQUE NOT NULL,
    paper_count INTEGER DEFAULT 0,
    h_index FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper-Author Junction Table (Many-to-Many)
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE,
    author_position INTEGER,  -- Order in author list
    PRIMARY KEY (paper_id, author_id)
);

-- Data Quality Metrics Table
-- KDD PHASE 2: Data Cleaning metrics
CREATE TABLE IF NOT EXISTS data_quality (
    quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_papers INTEGER,
    papers_with_abstract INTEGER,
    papers_with_citations INTEGER,
    average_abstract_length FLOAT,
    data_completeness FLOAT,  -- Percentage of non-null fields
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing Logs Table
-- Track all data mining operations
CREATE TABLE IF NOT EXISTS processing_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation VARCHAR(100),
    status VARCHAR(50),  -- 'SUCCESS' or 'FAILED'
    papers_processed INTEGER,
    execution_time_seconds FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature Extraction Table
-- KDD PHASE 4: Data Transformation
CREATE TABLE IF NOT EXISTS paper_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    tfidf_vector TEXT,  -- JSON serialized TF-IDF vector
    embedding_vector TEXT,  -- JSON serialized sentence embedding
    keyword_frequency TEXT,  -- JSON of top keywords
    feature_extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for Performance Optimization
CREATE INDEX IF NOT EXISTS idx_papers_publication_date ON papers(publication_date);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_paper_topics_topic_id ON paper_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_paper_topics_probability ON paper_topics(probability);
CREATE INDEX IF NOT EXISTS idx_paper_clusters_cluster_id ON paper_clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_trends_topic_id ON trends(topic_id);
CREATE INDEX IF NOT EXISTS idx_trends_year_month ON trends(year_month);
CREATE INDEX IF NOT EXISTS idx_forecasts_topic_id ON forecasts(topic_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_forecast_date ON forecasts(forecast_date);

"""

# Entity-Relationship Diagram Description
ER_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI Research Trend Mining - ER Diagram                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│    PAPERS        │ (Core Entity)
├──────────────────┤
│ paper_id (PK)    │
│ arxiv_id         │
│ semantic_id      │
│ title            │
│ abstract         │
│ authors          │
│ pub_date         │
│ citation_count   │
└──────────────────┘
       │ N-M
       ├──────────┬──────────┬──────────┬──────────┐
       │          │          │          │          │
    (1:N)      (1:N)      (1:N)      (1:N)      (1:N)
       │          │          │          │          │
       │          │          │          │          │
┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│  CATEGORIES │ │ PAPER_TOPICS │ │PAPER_CLUSTERS│ │   TOPICS   │ │  CLUSTERS    │
├─────────────┤ ├──────────────┤ ├──────────────┤ ├─────────────┤ ├──────────────┤
│ cat_id (PK) │ │ paper_id(FK) │ │ paper_id(FK) │ │ topic_id(PK)│ │cluster_id(PK)│
│ name        │ │ topic_id(FK) │ │ cluster_id(FK)│ │ keywords   │ │ label        │
│ description │ │ probability  │ │ distance     │ │ coherence  │ │ size         │
└─────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ └──────────────┘
                       │                               │ N-1
                       │                               │
                    (1:N)                           (1:N)
                       │                               │
                  ┌────────────┐                  ┌──────────────┐
                  │   TRENDS   │                  │  FORECASTS   │
                  ├────────────┤                  ├──────────────┤
                  │ trend_id(PK)│                  │forecast_id(PK)│
                  │ topic_id(FK)│                  │topic_id(FK)   │
                  │ year_month  │                  │forecast_date  │
                  │ frequency   │                  │prediction     │
                  │ growth_rate │                  │confidence     │
                  └────────────┘                  └──────────────┘

DATA MINING PHASES MAPPING:
─────────────────────────────
Phase 2 (Data Cleaning):     data_quality, processing_logs tables
Phase 3 (Data Integration):  All core tables, relationships
Phase 4 (Data Transformation): paper_features, paper_topics (embeddings)
Phase 5 (Pattern Discovery):  topics, clusters, paper_topics, paper_clusters
Phase 6 (Knowledge Discovery): trends table
Phase 7 (Predictive Mining):  forecasts table
"""

INSERT_SAMPLE_DATA = """
-- Insert sample categories
INSERT INTO categories (name, description) VALUES
    ('Artificial Intelligence', 'AI and machine learning research'),
    ('Natural Language Processing', 'NLP and text processing'),
    ('Computer Vision', 'Image and video analysis'),
    ('Deep Learning', 'Neural networks and deep architectures'),
    ('Reinforcement Learning', 'Learning through interaction'),
    ('Federated Learning', 'Distributed machine learning'),
    ('Large Language Models', 'LLMs and foundation models'),
    ('Graph Neural Networks', 'GNNs and graph learning'),
    ('Interpretability', 'AI explainability and interpretability'),
    ('Robotics', 'Robotic systems and automation');

-- Insert sample data quality record
INSERT INTO data_quality (total_papers, papers_with_abstract, papers_with_citations, average_abstract_length, data_completeness)
VALUES (0, 0, 0, 0.0, 0.0);
"""
