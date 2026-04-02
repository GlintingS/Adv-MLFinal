import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path):
    project_root = Path(__file__).resolve().parents[2]

    # Import the data from raw data file
    logger.info("Loading raw data from %s", data_path)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse CSV file: %s", exc)
        raise

    required_columns = [
        "price",
        "year_sold",
        "year_built",
        "property_type",
        "beds",
        "baths",
        "basement",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error("Missing required columns in dataset: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    # store the target variable in y
    y = df["price"]

    df["age"] = df["year_sold"] - df["year_built"]

    # seperate input features in x
    x = df.drop(
        [
            "price",
            "property_type",
            "year_sold",
            "year_built",
            "beds",
            "baths",
            "basement",
        ],
        axis=1,
    )

    output_path = project_root / "data" / "processed" / "cleaned_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        logger.info("Saved cleaned data to %s", output_path)
    except OSError as exc:
        logger.error("Failed to save cleaned data: %s", exc)
        raise

    return df, x, y
