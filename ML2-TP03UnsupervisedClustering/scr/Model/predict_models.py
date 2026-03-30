import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "kmodel.pkl"


def load_model():
    """Load the saved KMeans model from disk."""
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def predict_clusters(model, X):
    """Assign cluster labels to feature matrix X."""
    return model.predict(X)
