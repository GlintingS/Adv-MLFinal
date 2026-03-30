import pandas as pd
from pathlib import Path


def load_and_preprocess_data(data_path):
    project_root = Path(__file__).resolve().parents[2]

    # Import the data from raw data file
    df = pd.read_csv(data_path)

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
    df.to_csv(output_path, index=False)

    return df, x, y
