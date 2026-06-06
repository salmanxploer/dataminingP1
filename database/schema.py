"""
Database Schema Definition as Python Strings
KDD PHASE 3: Data Integration
"""

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id VARCHAR(50) UNIQUE,
    semantic_scholar_id VARCHAR(100) UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    publication_date DATE,
    source VARCHAR(50),
    url TEXT UNIQUE,
    pdf_url TEXT,
    citation_count INTEGER DEFAULT 0,
    influence_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parent_category_id INTEGER REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS paper_categories (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(category_id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, category_id)
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_label VARCHAR(100),
    top_keywords TEXT,
    topic_description TEXT,
    algorithm VARCHAR(50),
    coherence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    probability FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (paper_id, topic_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_label INTEGER,
    cluster_name VARCHAR(100),
    size INTEGER,
    algorithm VARCHAR(50),
    silhouette_score FLOAT,
    davies_bouldin_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_clusters (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    distance_to_center FLOAT,
    PRIMARY KEY (paper_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS trends (
    trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    year_month VARCHAR(7),
    frequency INTEGER,
    growth_rate FLOAT,
    emerging_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    forecast_date DATE,
    predicted_frequency FLOAT,
    lower_bound FLOAT,
    upper_bound FLOAT,
    confidence_interval FLOAT,
    forecast_model VARCHAR(50),
    mape FLOAT,
    rmse FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) UNIQUE NOT NULL,
    paper_count INTEGER DEFAULT 0,
    h_index FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE,
    author_position INTEGER,
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE IF NOT EXISTS data_quality (
    quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_papers INTEGER,
    papers_with_abstract INTEGER,
    papers_with_citations INTEGER,
    average_abstract_length FLOAT,
    data_completeness FLOAT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processing_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation VARCHAR(100),
    status VARCHAR(50),
    papers_processed INTEGER,
    execution_time_seconds FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    tfidf_vector TEXT,
    embedding_vector TEXT,
    keyword_frequency TEXT,
    feature_extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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

INSERT_SAMPLE_DATA = """
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

INSERT INTO data_quality (total_papers, papers_with_abstract, papers_with_citations, average_abstract_length, data_completeness)
VALUES (0, 0, 0, 0.0, 0.0);
"""
