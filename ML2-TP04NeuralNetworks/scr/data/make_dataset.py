from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Admit_Chance"
FEATURE_COLUMNS = [
    "GRE_Score",
    "TOEFL_Score",
    "University_Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]
CATEGORICAL_COLUMNS = ["University_Rating", "Research"]


def load_admission_data(data_path: str | Path) -> pd.DataFrame:
    """Load UCLA admission data from CSV."""
    return pd.read_csv(data_path)


def prepare_features_target(
    df: pd.DataFrame,
    admit_threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a modeling frame for classification.

    Returns `(X, y, prepared_df)` where:
    - `X` contains feature columns only
    - `y` is the binary target (0/1)
    - `prepared_df` contains all cleaned fields including target
    """
    prepared = df.copy()

    if "Serial_No" in prepared.columns:
        prepared = prepared.drop(columns=["Serial_No"])

    prepared[TARGET_COLUMN] = (prepared[TARGET_COLUMN] >= admit_threshold).astype(int)

    for col in CATEGORICAL_COLUMNS:
        prepared[col] = prepared[col].astype(str)

    X = prepared[FEATURE_COLUMNS].copy()
    y = prepared[TARGET_COLUMN].copy()
    return X, y, prepared


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/test split with stratification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_prepare_split(
    data_path: str | Path,
    admit_threshold: float = 0.80,
    test_size: float = 0.2,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Convenience function for loading, cleaning, and splitting the dataset."""
    raw_df = load_admission_data(data_path)
    X, y, prepared_df = prepare_features_target(raw_df, admit_threshold=admit_threshold)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, prepared_df
