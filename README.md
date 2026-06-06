# AI Research Trend Mining and Emerging Topic Discovery System

## 📋 Project Overview

This is a **comprehensive Data Mining project** designed for university-level coursework. The system automatically collects AI research papers from multiple sources, applies advanced data mining techniques to discover hidden patterns, identifies emerging research trends, and provides interactive visualizations through a Streamlit dashboard.

### Project Goal
Build a complete Data Mining pipeline that demonstrates all core phases of the KDD (Knowledge Discovery in Databases) process:
- **Data Collection**: Harvest papers from arXiv and Semantic Scholar APIs
- **Data Cleaning**: Text preprocessing and normalization
- **Data Integration**: Database consolidation
- **Data Transformation**: Feature extraction and embedding generation
- **Pattern Discovery**: Topic modeling (LDA, BERTopic)
- **Knowledge Discovery**: Clustering and trend analysis
- **Predictive Mining**: Future trend forecasting

---

## 🎯 Key Objectives

✅ Collect 10,000+ AI research papers with metadata  
✅ Implement LDA and BERTopic for topic modeling  
✅ Apply HDBSCAN and K-Means clustering  
✅ Perform trend analysis and emerging topic detection  
✅ Forecast future research directions  
✅ Create interactive dashboard with 6+ pages  
✅ Generate 8+ advanced visualizations  
✅ Document all data mining phases  

---

## 📊 Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **Web Framework** | Streamlit |
| **Database** | SQLite / PostgreSQL |
| **Data Processing** | Pandas, NumPy |
| **ML/Mining** | Scikit-Learn, Gensim, BERTopic |
| **NLP** | NLTK, SpaCy, Sentence Transformers |
| **Visualization** | Plotly, Matplotlib |
| **Forecasting** | Prophet, ARIMA |
| **ORM** | SQLAlchemy |

---

## 📁 Project Structure

```
dataminingP1/
│
├── 📂 data/
│   ├── raw/                          # Raw API responses
│   ├── processed/                    # Cleaned datasets
│   └── samples/                      # Sample datasets
│
├── 📂 database/
│   ├── schema.sql                    # Database schema
│   ├── ER_diagram.md                 # ER diagram explanation
│   └── init_db.py                    # Database initialization
│
├── 📂 src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   │
│   ├── 📂 data_collection/
│   │   ├── __init__.py
│   │   ├── arxiv_collector.py        # arXiv API integration
│   │   ├── semantic_scholar.py       # Semantic Scholar API
│   │   └── db_manager.py             # Database operations
│   │
│   ├── 📂 preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py           # Text cleaning utilities
│   │   ├── preprocess_pipeline.py    # Complete pipeline
│   │   └── tokenizer.py              # Tokenization & lemmatization
│   │
│   ├── 📂 topic_modeling/
│   │   ├── __init__.py
│   │   ├── lda_model.py              # LDA implementation
│   │   ├── bertopic_model.py         # BERTopic implementation
│   │   └── topic_evaluator.py        # Coherence & metrics
│   │
│   ├── 📂 clustering/
│   │   ├── __init__.py
│   │   ├── kmeans_cluster.py         # K-Means clustering
│   │   ├── hdbscan_cluster.py        # HDBSCAN clustering
│   │   └── cluster_evaluator.py      # Silhouette, Davies-Bouldin
│   │
│   ├── 📂 trend_analysis/
│   │   ├── __init__.py
│   │   ├── trend_analyzer.py         # Trend calculation
│   │   ├── emerging_topics.py        # Emerging topic detection
│   │   └── growth_metrics.py         # Growth rate analysis
│   │
│   ├── 📂 forecasting/
│   │   ├── __init__.py
│   │   ├── prophet_forecaster.py     # Prophet forecasting
│   │   ├── arima_forecaster.py       # ARIMA forecasting
│   │   └── forecast_evaluator.py     # Forecast metrics
│   │
│   ├── 📂 visualization/
│   │   ├── __init__.py
│   │   ├── plotly_charts.py          # Plotly visualizations
│   │   ├── matplotlib_charts.py      # Matplotlib charts
│   │   ├── wordcloud_gen.py          # Word cloud generation
│   │   └── network_graph.py          # Network visualization
│   │
│   └── 📂 dashboard/
│       ├── __init__.py
│       ├── app.py                    # Main dashboard app
│       ├── pages/
│       │   ├── 1_dataset_overview.py
│       │   ├── 2_topic_explorer.py
│       │   ├── 3_cluster_explorer.py
│       │   ├── 4_trend_analytics.py
│       │   ├── 5_forecast_dashboard.py
│       │   └── 6_emerging_topics.py
│       └── utils.py
│
├── 📂 notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_clustering_analysis.ipynb
│   ├── 05_trend_analysis.ipynb
│   ├── 06_forecasting.ipynb
│   └── 07_integrated_workflow.ipynb
│
├── 📂 reports/
│   ├── DATA_MINING_PHASES.md         # KDD process documentation
│   ├── ARCHITECTURE.md               # System architecture
│   ├── METHODOLOGY.md                # Research methodology
│   ├── EVALUATION.md                 # Model evaluation results
│   ├── FINDINGS.md                   # Key findings & insights
│   └── PRESENTATION_OUTLINE.md       # PowerPoint structure
│
├── 📂 models/
│   ├── lda_model.pkl                 # Trained LDA
│   ├── bertopic_model.pkl            # Trained BERTopic
│   ├── kmeans_model.pkl              # Trained K-Means
│   └── prophet_models.pkl            # Prophet models
│
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── .env.example                      # Environment variables template
└── LICENSE                           # MIT License
```

---

## 🔄 Data Mining Phases (KDD Process)

This project demonstrates all 6 phases of the KDD (Knowledge Discovery in Databases) process:

### Phase 1: Data Collection 📥
**Location**: `src/data_collection/`  
**Methods**: arXiv API, Semantic Scholar API  
**Output**: Raw paper metadata  

### Phase 2: Data Cleaning 🧹
**Location**: `src/preprocessing/text_cleaner.py`  
**Operations**: Lowercasing, stopword removal, noise removal  
**Output**: Cleaned text corpus  

### Phase 3: Data Integration 🔗
**Location**: `src/data_collection/db_manager.py`  
**Methods**: Database schema design, data consolidation  
**Output**: Unified database  

### Phase 4: Data Transformation 🔄
**Location**: `src/preprocessing/preprocess_pipeline.py`  
**Operations**: Tokenization, lemmatization, TF-IDF, embeddings  
**Output**: Feature vectors  

### Phase 5: Pattern Discovery & Knowledge Discovery 🔎
**Location**: `src/topic_modeling/`, `src/clustering/`  
**Methods**: LDA, BERTopic, K-Means, HDBSCAN  
**Output**: Topics, clusters, patterns  

### Phase 6: Predictive Data Mining 🔮
**Location**: `src/forecasting/`  
**Methods**: Prophet, ARIMA time-series forecasting  
**Output**: Future trend predictions  

---

## 📊 Database Schema

### Core Tables

**papers**
- paper_id (PK)
- title
- authors
- abstract
- publication_date
- source (arxiv/semantic_scholar)
- url
- citation_count
- created_at

**categories**
- category_id (PK)
- name
- description

**paper_categories**
- paper_id (FK)
- category_id (FK)

**topics**
- topic_id (PK)
- topic_name
- top_keywords
- created_at

**paper_topics**
- paper_id (FK)
- topic_id (FK)
- probability

**clusters**
- cluster_id (PK)
- cluster_label
- size
- created_at

**paper_clusters**
- paper_id (FK)
- cluster_id (FK)

**trends**
- trend_id (PK)
- topic_id (FK)
- month
- frequency
- growth_rate

**forecasts**
- forecast_id (PK)
- topic_id (FK)
- forecast_date
- predicted_frequency
- confidence_interval

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/salmanxploer/dataminingP1.git
cd dataminingP1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Initialize Database

```bash
python src/database/init_db.py
```

### Collect Data

```bash
python -m src.data_collection.arxiv_collector
python -m src.data_collection.semantic_scholar
```

### Run Preprocessing

```bash
python -m src.preprocessing.preprocess_pipeline
```

### Train Models

```bash
python -m src.topic_modeling.lda_model
python -m src.topic_modeling.bertopic_model
python -m src.clustering.kmeans_cluster
```

### Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## 📈 Expected Results

- **10,000+ papers** collected from APIs
- **50-100 topics** identified via topic modeling
- **5-10 clusters** discovered via clustering
- **Coherence score > 0.5** for topic models
- **Silhouette score > 0.4** for clusters
- **3-6 month forecasts** for trend prediction

---

## 📖 Documentation

Comprehensive documentation is provided in the `reports/` directory:

- **DATA_MINING_PHASES.md**: Detailed explanation of KDD process
- **ARCHITECTURE.md**: System design and components
- **METHODOLOGY.md**: Research methodology and approaches
- **EVALUATION.md**: Model performance metrics and comparisons
- **FINDINGS.md**: Key insights and discoveries

---

## 📊 Dashboard Pages

1. **Dataset Overview** - Summary statistics and data quality metrics
2. **Topic Explorer** - Interactive topic analysis and word clouds
3. **Cluster Explorer** - Cluster visualization and analysis
4. **Trend Analytics** - Historical trend evolution and growth rates
5. **Forecast Dashboard** - Future trend predictions with confidence intervals
6. **Emerging Topics** - Detection and analysis of new research areas

---

## 🎓 Learning Outcomes

By completing this project, you will understand:

✅ Data collection from multiple APIs  
✅ Text preprocessing and NLP pipelines  
✅ Topic modeling (LDA vs BERTopic)  
✅ Unsupervised clustering algorithms  
✅ Time-series analysis and forecasting  
✅ Data mining evaluation metrics  
✅ Interactive dashboard development  
✅ Database design and management  
✅ End-to-end ML/mining workflows  

---

## 📚 References

- Blei et al. (2003) - Latent Dirichlet Allocation
- Grootendorst (2022) - BERTopic: Topic Modeling
- Campello et al. (2015) - HDBSCAN: Hierarchical Clustering
- Taylor & Letham (2018) - Prophet: Forecasting
- Röder et al. (2015) - Evaluating Topic Models

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Salman Exploer**  
Data Mining Research Project  
University Capstone

---

## 📞 Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Last Updated**: June 2026  
**Status**: In Development  
**Completeness**: 0% → 100%

