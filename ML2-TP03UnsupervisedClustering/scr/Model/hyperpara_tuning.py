import logging

from Model.train_models import train_kmeans  # noqa: F401  (re-exported for convenience)

logger = logging.getLogger(__name__)


def tune_kmeans(X, n_clusters=5, init="k-means++", n_init="auto", max_iter=300):
    """
    Train a KMeans model with explicit hyperparameters.

    Parameters
    ----------
    X          : feature matrix
    n_clusters : number of clusters
    init       : centroid initialisation strategy ('k-means++' or 'random')
    n_init     : number of times KMeans is run with different centroid seeds
    max_iter   : maximum number of EM iterations
    """
    from sklearn.cluster import KMeans
    import pickle
    from pathlib import Path

    logger.info(
        "Tuning KMeans: k=%d, init=%s, n_init=%s, max_iter=%d",
        n_clusters,
        init,
        n_init,
        max_iter,
    )
    try:
        kmodel = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42,
        ).fit(X)
    except Exception as exc:
        logger.error("KMeans tuning failed: %s", exc)
        raise

    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "kmodel.pkl"
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as f:
            pickle.dump(kmodel, f)
        logger.info("Tuned model saved to %s", model_path)
    except OSError as exc:
        logger.warning("Could not save tuned model to %s: %s", model_path, exc)

    return kmodel
