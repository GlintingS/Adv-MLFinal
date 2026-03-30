from __future__ import annotations

from pathlib import Path
import pickle
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from scr.data.make_dataset import CATEGORICAL_COLUMNS


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
    model = build_mlp_pipeline(X_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sklearn.utils.extmath",
        )
        model.fit(X_train, y_train)
    return model


def save_model(model: Pipeline, model_path: str | Path) -> Path:
    """Persist a trained model to disk."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)
    return model_path


def load_model(model_path: str | Path) -> Pipeline:
    """Load a persisted model from disk."""
    with Path(model_path).open("rb") as model_file:
        return pickle.load(model_file)
