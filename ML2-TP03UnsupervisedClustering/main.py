import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Make scr/ importable when running main.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "scr"))

from data import make_dataset
from Model import train_models, predict_models
from visuals import visualize

FEATURES_2D = ["Annual_Income", "Spending_Score"]
FEATURES_3D = ["Age", "Annual_Income", "Spending_Score"]

if __name__ == "__main__":
    # Load data
    df = make_dataset.load_data()
    print(f"Loaded {len(df)} rows  |  columns: {list(df.columns)}")
    print(df.describe())

    # ── 2-feature analysis ────────────────────────────────────────────────
    X2 = make_dataset.get_features(df, FEATURES_2D)

    wss_df = train_models.run_elbow_method(X2)
    print("\nElbow (2-feature):\n", wss_df)
    fig_e = visualize.plot_elbow(wss_df)
    plt.show()

    sil_df = train_models.run_silhouette(X2)
    print("\nSilhouette (2-feature):\n", sil_df)
    fig_s = visualize.plot_silhouette(sil_df)
    plt.show()

    # Train final model with optimal k=5
    kmodel = train_models.train_kmeans(X2, n_clusters=5)
    df["Cluster"] = predict_models.predict_clusters(kmodel, X2)
    print("\nCluster distribution:\n", df["Cluster"].value_counts())

    centers = kmodel.cluster_centers_
    fig_c = visualize.plot_clusters(
        df, "Annual_Income", "Spending_Score", "Cluster", centers
    )
    plt.show()

    # ── 3-feature analysis ────────────────────────────────────────────────
    X3 = make_dataset.get_features(df, FEATURES_3D)

    wss_df3 = train_models.run_elbow_method(X3)
    print("\nElbow (3-feature):\n", wss_df3)

    sil_df3 = train_models.run_silhouette(X3)
    print("\nSilhouette (3-feature):\n", sil_df3)
