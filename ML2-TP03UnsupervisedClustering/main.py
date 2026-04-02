import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ── Logging configuration ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mall_clustering.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Make scr/ importable when running main.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "scr"))

from data import make_dataset
from Model import train_models, predict_models
from visuals import visualize

FEATURES_2D = ["Annual_Income", "Spending_Score"]
FEATURES_3D = ["Age", "Annual_Income", "Spending_Score"]

if __name__ == "__main__":
    try:
        # Load data
        df = make_dataset.load_data()
        logger.info("Loaded %d rows  |  columns: %s", len(df), list(df.columns))
        print(df.describe())

        # ── 2-feature analysis ────────────────────────────────────────────
        X2 = make_dataset.get_features(df, FEATURES_2D)

        wss_df = train_models.run_elbow_method(X2)
        logger.info("Elbow (2-feature) computed")
        print("\nElbow (2-feature):\n", wss_df)
        fig_e = visualize.plot_elbow(wss_df)
        plt.show()

        sil_df = train_models.run_silhouette(X2)
        logger.info("Silhouette (2-feature) computed")
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

        # ── 3-feature analysis ────────────────────────────────────────────
        X3 = make_dataset.get_features(df, FEATURES_3D)

        wss_df3 = train_models.run_elbow_method(X3)
        logger.info("Elbow (3-feature) computed")
        print("\nElbow (3-feature):\n", wss_df3)

        sil_df3 = train_models.run_silhouette(X3)
        logger.info("Silhouette (3-feature) computed")
        print("\nSilhouette (3-feature):\n", sil_df3)

        logger.info("Pipeline completed successfully")

    except FileNotFoundError as exc:
        logger.error("Data file error: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        logger.error("Column error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)
