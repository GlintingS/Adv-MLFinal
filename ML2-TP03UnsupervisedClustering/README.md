# Mall Customer Segmentation — Unsupervised Clustering

An unsupervised K-Means clustering project that segments mall customers based on
**Annual Income**, **Spending Score**, and optionally **Age**. It includes an
interactive Streamlit dashboard and a CLI pipeline.

## Project Structure

```
├── main.py                                  # CLI pipeline (elbow, silhouette, clustering)
├── streamlit_03UnsupervisedClustering_app.py # Streamlit web app
├── verify.py                                # Pre-flight readiness checker
├── requirements.txt                         # Python dependencies
├── runtime.txt                              # Python version (3.11.9)
├── mall_customers.csv                       # Dataset (fallback location)
├── data/
│   └── raw/
│       └── mall_customers.csv               # Primary dataset location
├── models/                                  # Saved KMeans model (.pkl)
└── scr/
    ├── data/
    │   └── make_dataset.py                  # Data loading & feature extraction
    ├── Model/
    │   ├── train_models.py                  # Elbow, silhouette, KMeans training
    │   ├── predict_models.py                # Cluster prediction & model loading
    │   └── hyperpara_tuning.py              # Hyperparameter tuning wrapper
    └── visuals/
        └── visualize.py                     # Plotting (clusters, elbow, silhouette, pairplot)
```

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - streamlit ≥ 1.35
  - pandas ≥ 2.0
  - numpy ≥ 1.26
  - scikit-learn ≥ 1.4
  - matplotlib ≥ 3.8
  - seaborn ≥ 0.13

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

### Verify environment

```bash
python verify.py
```

This checks Python version, installed packages, project structure, and data file
availability. Fix any `[FAIL]` items before proceeding.

### CLI pipeline

```bash
python main.py
```

Runs the full analysis (elbow method, silhouette scores, K-Means clustering) and
displays matplotlib plots. Logs are written to `mall_clustering.log`.

### Streamlit web app

```bash
streamlit run streamlit_03UnsupervisedClustering_app.py
```

Opens an interactive dashboard with tabs for data overview, elbow plot,
silhouette plot, and cluster visualisation. Sidebar controls let you switch
between 2D/3D feature sets and adjust the number of clusters.

## Dataset

`mall_customers.csv` — 200 rows with the following columns:

| Column         | Description                      |
|----------------|----------------------------------|
| CustomerID     | Unique customer identifier       |
| Gender         | Male / Female                    |
| Age            | Customer age                     |
| Annual_Income  | Annual income (k$)               |
| Spending_Score | Spending score assigned (1–100)  |

Place the file in `data/raw/` (preferred) or the project root.

## Logging

All modules use Python's `logging` library. When running via `main.py` or the
Streamlit app, logs are output to the console and appended to
`mall_clustering.log` in the project root.
