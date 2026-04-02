import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "kmodel.pkl"


def load_model():
    """Load the saved KMeans model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from %s", MODEL_PATH)
        return model
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
        raise


def predict_clusters(model, X):
    """Assign cluster labels to feature matrix X."""
    try:
        labels = model.predict(X)
        logger.info(
            "Predicted %d samples into %d clusters", len(labels), len(set(labels))
        )
        return labels
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise
