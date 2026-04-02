import logging

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_clusters(df, x_col, y_col, cluster_col, centers):
    """Scatter plot of two features coloured by cluster, with centroids marked."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x=x_col, y=y_col, data=df, hue=cluster_col, palette="colorblind", ax=ax
        )
        ax.scatter(
            centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.6, label="Centroids"
        )
        ax.set_title("Customer Clusters")
        ax.legend()
        logger.info("Cluster scatter plot created")
        return fig
    except Exception as exc:
        logger.error("Failed to create cluster plot: %s", exc)
        raise


def plot_elbow(wss_df):
    """Line plot of WCSS vs. number of clusters (Elbow Plot)."""
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(wss_df["cluster"], wss_df["WCSS_Score"], marker="o")
        ax.set_xlabel("No. of Clusters")
        ax.set_ylabel("WCSS Score")
        ax.set_title("Elbow Plot")
        ax.grid(True)
        logger.info("Elbow plot created")
        return fig
    except Exception as exc:
        logger.error("Failed to create elbow plot: %s", exc)
        raise


def plot_silhouette(sil_df):
    """Line plot of Silhouette Score vs. number of clusters."""
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            sil_df["cluster"], sil_df["Silhouette_Score"], marker="o", color="orange"
        )
        ax.set_xlabel("No. of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Plot")
        ax.grid(True)
        logger.info("Silhouette plot created")
        return fig
    except Exception as exc:
        logger.error("Failed to create silhouette plot: %s", exc)
        raise


def plot_pairplot(df, cols):
    """Seaborn pairplot for the given columns."""
    try:
        grid = sns.pairplot(df[cols])
        logger.info("Pair plot created")
        return grid.fig
    except Exception as exc:
        logger.error("Failed to create pair plot: %s", exc)
        raise
