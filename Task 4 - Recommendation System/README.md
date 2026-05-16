# 🎬 CineMatch — Hybrid Movie Recommendation System

**CodSoft AI Internship — Task 4**

A smart movie recommendation engine that combines **Content-Based Filtering** and **Collaborative Filtering** into an explainable hybrid system. Built as part of the **CodSoft AI Internship — Task 4**.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Content-Based Filtering** | TF-IDF vectorisation on genres, director, cast & plot with cosine similarity |
| **Collaborative Filtering** | Truncated SVD (k=15) matrix factorisation on user-item ratings |
| **Hybrid Engine** | Adjustable α-weighted blend of both approaches |
| **Explainable Picks** | Every recommendation includes a human-readable explanation |
| **Interactive Analytics** | Genre radar charts, similarity heatmaps, rating distributions |
| **Movie Explorer** | Search, filter by genre, and discover similar titles |

## 🏗️ Architecture

```
recommendation-system/
├── app.py                   # Streamlit UI
├── engine/
│   ├── __init__.py
│   ├── data_manager.py      # Data loading & synthetic rating generation
│   ├── content_based.py     # TF-IDF + Cosine Similarity engine
│   ├── collaborative.py     # SVD matrix factorisation engine
│   └── hybrid.py            # Weighted hybrid blender
├── data/
│   ├── movies.csv           # 50-movie curated catalogue
│   └── ratings.csv          # Auto-generated user ratings
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## 🧠 Algorithms

### Content-Based Filtering
- Combines movie metadata (genres, director, cast, overview) into a single text document per movie
- Applies **TF-IDF** vectorisation with bi-gram support (up to 5,000 features)
- Computes pairwise **cosine similarity** across the full catalogue
- Recommends movies most similar to the user's highly-rated titles

### Collaborative Filtering (SVD)
- Constructs a **user–item rating matrix** from observed ratings
- Applies **truncated SVD** decomposition to extract 15 latent factors
- Reconstructs the full matrix to predict unobserved ratings
- Evaluated with **RMSE** and **MAE** on a held-out test split

### Hybrid Engine
- Normalises scores from both engines to [0, 1]
- Linearly combines them: `hybrid = α × content + (1 − α) × collaborative`
- Adjustable via the sidebar slider in real time

## 📊 Evaluation Metrics

The system reports **RMSE**, **MAE**, and test sample count on the Analytics tab, computed via a 20% held-out evaluation split from the collaborative filtering model.

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **scikit-learn** — TF-IDF, cosine similarity
- **SciPy** — truncated SVD
- **Plotly** — interactive charts
- **Pandas / NumPy** — data processing
