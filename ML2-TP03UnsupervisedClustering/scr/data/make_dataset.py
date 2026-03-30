from pathlib import Path

import pandas as pd

# Canonical raw-data location inside the project
_RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "mall_customers.csv"
# Fallback: file sitting in the project root
_ROOT_PATH = Path(__file__).resolve().parents[2] / "mall_customers.csv"


def load_data(data_path=None):
    """Load mall_customers.csv and return the dataframe.

    Search order (first match wins):
    1. ``data_path`` argument, if provided
    2. data/raw/mall_customers.csv
    3. <project_root>/mall_customers.csv
    """
    candidates = []
    if data_path:
        candidates.append(Path(data_path))
    candidates += [_RAW_PATH, _ROOT_PATH]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)

    raise FileNotFoundError(
        "mall_customers.csv not found or is empty (OneDrive placeholder). "
        "Sync the file locally first."
    )


def get_features(df, feature_cols):
    """Extract a sub-dataframe containing only the requested feature columns."""
    return df[feature_cols]
