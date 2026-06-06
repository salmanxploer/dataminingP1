"""
Configuration management for AI Research Trend Mining System
Handles API keys, database connections, and model parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# API Configuration
ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_SEARCH_QUERY = "cat:cs.AI AND submittedDate:[202401010000 TO 202512312359]"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "10000"))
PAPERS_PER_REQUEST = int(os.getenv("PAPERS_PER_REQUEST", "100"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_research_mining.db")
ECHO_SQL = os.getenv("ECHO_SQL", "False").lower() == "true"

# Text Preprocessing Configuration
STOPWORDS_LANGUAGE = "english"
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 20
REMOVE_URLS = True
REMOVE_EMAILS = True
REMOVE_NUMBERS = False
LEMMATIZATION = True

# Topic Modeling Configuration
LDA_CONFIG = {
    "num_topics": 50,
    "passes": 10,
    "workers": 4,
    "per_word_topics": True,
    "minimum_probability": 0.0,
}

BERTOPIC_CONFIG = {
    "nr_topics": 50,
    "min_topic_size": 10,
    "language": "english",
    "calculate_probabilities": True,
}

# Clustering Configuration
KMEANS_CONFIG = {
    "n_clusters": 10,
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

HDBSCAN_CONFIG = {
    "min_cluster_size": 20,
    "min_samples": 10,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

# TF-IDF Configuration
TFIDF_CONFIG = {
    "max_df": 0.8,
    "min_df": 5,
    "max_features": 5000,
    "ngram_range": (1, 2),
}

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
EMBEDDING_DIM = 384

# Forecasting Configuration
PROPHET_CONFIG = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "interval_width": 0.95,
}

ARIMA_CONFIG = {
    "order": (1, 1, 1),  # (p, d, q)
    "seasonal_order": (0, 0, 0, 0),  # (P, D, Q, m)
}

FORECAST_MONTHS = 6

# Visualization Configuration
PLOTLY_THEME = "plotly_white"
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Evaluation Metrics Configuration
COHERENCE_METRIC = "c_v"  # c_v, u_mass, c_uci, c_npmi
MINIMUM_COHERENCE = 0.5
MINIMUM_SILHOUETTE = 0.3

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# Random Seeds for Reproducibility
RANDOM_SEED = 42

# Dashboard Configuration
DASHBOARD_THEME = "light"
MAX_DISPLAY_ITEMS = 100
