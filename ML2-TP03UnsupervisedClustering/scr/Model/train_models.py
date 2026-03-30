import pickle
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "kmodel.pkl"


def run_elbow_method(X, k_range=range(3, 9)):
    """Compute WCSS for each k and return a summary dataframe."""
    K, WCSS = [], []
    for i in k_range:
        kmodel = KMeans(
            n_clusters=i, init="k-means++", n_init="auto", random_state=42
        ).fit(X)
        WCSS.append(kmodel.inertia_)
        K.append(i)
    return pd.DataFrame({"cluster": K, "WCSS_Score": WCSS})


def run_silhouette(X, k_range=range(3, 9)):
    """Compute Silhouette score for each k and return a summary dataframe."""
    K, ss = [], []
    for i in k_range:
        kmodel = KMeans(
            n_clusters=i, init="k-means++", n_init="auto", random_state=42
        ).fit(X)
        sil_score = silhouette_score(X, kmodel.labels_)
        K.append(i)
        ss.append(sil_score)
    return pd.DataFrame({"cluster": K, "Silhouette_Score": ss})


def train_kmeans(X, n_clusters=5):
    """Train a KMeans model with the given number of clusters and save it to disk."""
    kmodel = KMeans(
        n_clusters=n_clusters, init="k-means++", n_init="auto", random_state=42
    ).fit(X)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(kmodel, f)
    return kmodel
