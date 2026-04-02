from __future__ import annotations

import logging
from pathlib import Path
import pickle
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from scr.data.make_dataset import CATEGORICAL_COLUMNS

logger = logging.getLogger(__name__)


def build_mlp_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build preprocessing + neural network pipeline."""
    numeric_columns = [c for c in X_train.columns if c not in CATEGORICAL_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        solver="lbfgs",
        alpha=0.0001,
        max_iter=3000,
        random_state=123,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_mlp_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train the neural network model."""
    logger.info("Training MLP model on %d samples", len(X_train))
    model = build_mlp_pipeline(X_train)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module="sklearn.utils.extmath",
            )
            model.fit(X_train, y_train)
    except Exception:
        logger.exception("Model training failed")
        raise
    logger.info("Model training completed successfully")
    return model


def save_model(model: Pipeline, model_path: str | Path) -> Path:
    """Persist a trained model to disk."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file)
    except Exception:
        logger.exception("Failed to save model to %s", model_path)
        raise
    logger.info("Model saved to %s", model_path)
    return model_path


def load_model(model_path: str | Path) -> Pipeline:
    """Load a persisted model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        with model_path.open("rb") as model_file:
            model = pickle.load(model_file)
    except Exception:
        logger.exception("Failed to load model from %s", model_path)
        raise
    logger.info("Model loaded from %s", model_path)
    return model
