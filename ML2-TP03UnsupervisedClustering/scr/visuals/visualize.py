import matplotlib.pyplot as plt
import seaborn as sns


def plot_clusters(df, x_col, y_col, cluster_col, centers):
    """Scatter plot of two features coloured by cluster, with centroids marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=x_col, y=y_col, data=df, hue=cluster_col, palette="colorblind", ax=ax
    )
    ax.scatter(
        centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.6, label="Centroids"
    )
    ax.set_title("Customer Clusters")
    ax.legend()
    return fig


def plot_elbow(wss_df):
    """Line plot of WCSS vs. number of clusters (Elbow Plot)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wss_df["cluster"], wss_df["WCSS_Score"], marker="o")
    ax.set_xlabel("No. of Clusters")
    ax.set_ylabel("WCSS Score")
    ax.set_title("Elbow Plot")
    ax.grid(True)
    return fig


def plot_silhouette(sil_df):
    """Line plot of Silhouette Score vs. number of clusters."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sil_df["cluster"], sil_df["Silhouette_Score"], marker="o", color="orange")
    ax.set_xlabel("No. of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Plot")
    ax.grid(True)
    return fig


def plot_pairplot(df, cols):
    """Seaborn pairplot for the given columns."""
    grid = sns.pairplot(df[cols])
    return grid.fig
