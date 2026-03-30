from Model.train_models import train_kmeans  # noqa: F401  (re-exported for convenience)


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

    kmodel = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=42,
    ).fit(X)

    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "kmodel.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(kmodel, f)

    return kmodel
