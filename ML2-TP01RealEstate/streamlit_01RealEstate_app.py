import logging
from pathlib import Path
import pickle
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "RFmodel.pkl"
FEATURE_IMPORTANCE_IMAGE = PROJECT_ROOT / "feature_importance.png"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "real_estate.csv"
FEATURE_COLUMNS = ["property_tax", "insurance", "sqft", "lot_size", "age"]

st.set_page_config(page_title="Real Estate Price Predictor", page_icon="house")

# Set the page title and description
st.title("Real Estate Price Predictor")
st.write(
    """
This app predicts real estate price using your trained Random Forest model.
Enter property details below to estimate the sale price.
"""
)

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()


@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    return pickle.loads(model_bytes)


def get_model():
    if MODEL_PATH.exists():
        logger.info("Loading model from %s", MODEL_PATH)
        return (
            load_model_from_bytes(MODEL_PATH.read_bytes()),
            f"Loaded model from {MODEL_PATH}",
        )

    st.warning(
        "No local model found at models/RFmodel.pkl. Upload a trained model file to continue."
    )
    uploaded_model = st.file_uploader("Upload Random Forest model (.pkl)", type=["pkl"])
    if uploaded_model is None:
        return None, "Model not loaded"

    try:
        return load_model_from_bytes(uploaded_model.getvalue()), "Loaded uploaded model"
    except Exception as exc:
        logger.error("Could not read uploaded model file: %s", exc)
        st.error(f"Could not read uploaded model file: {exc}")
        return None, "Failed to load uploaded model"


@st.cache_data
def load_reference_data() -> tuple[Optional[pd.DataFrame], str]:
    if PROCESSED_DATA_PATH.exists():
        return (
            pd.read_csv(PROCESSED_DATA_PATH),
            f"Loaded data from {PROCESSED_DATA_PATH}",
        )
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH), f"Loaded data from {RAW_DATA_PATH}"
    return None, "No local dataset found for charts"


def build_evaluation_matrix(
    df: pd.DataFrame, model
) -> tuple[Optional[pd.DataFrame], str]:
    needed_columns = FEATURE_COLUMNS + ["price"]
    missing_columns = [c for c in needed_columns if c not in df.columns]
    if missing_columns:
        return None, f"Missing required columns for evaluation: {missing_columns}"

    eval_df = df[needed_columns].dropna()
    if len(eval_df) < 50:
        return None, "Not enough valid rows to compute a stable evaluation matrix"

    x = eval_df[FEATURE_COLUMNS]
    y = eval_df["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    metric_rows = []

    if model is not None and hasattr(model, "predict"):
        y_pred_rf = model.predict(x_test)
        rf_rmse = mean_squared_error(y_test, y_pred_rf) ** 0.5
        metric_rows.append(
            {
                "Model": "Random Forest (loaded)",
                "MAE": mean_absolute_error(y_test, y_pred_rf),
                "RMSE": rf_rmse,
                "R2": r2_score(y_test, y_pred_rf),
            }
        )

    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    y_pred_lr = lr_model.predict(x_test)
    lr_rmse = mean_squared_error(y_test, y_pred_lr) ** 0.5
    metric_rows.append(
        {
            "Model": "Linear Regression (baseline)",
            "MAE": mean_absolute_error(y_test, y_pred_lr),
            "RMSE": lr_rmse,
            "R2": r2_score(y_test, y_pred_lr),
        }
    )

    matrix_df = pd.DataFrame(metric_rows)
    return matrix_df, f"Computed on {len(eval_df)} rows (80/20 split)"


rf_model, model_message = get_model()
st.caption(model_message)

reference_df, data_message = load_reference_data()
st.caption(data_message)


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Property Details")

    property_tax = st.number_input("Property Tax", min_value=0, step=10)
    insurance = st.number_input("Insurance", min_value=0, step=10)
    sqft = st.number_input("Square Footage (sqft)", min_value=100, step=10)
    lot_size = st.number_input("Lot Size", min_value=0, step=100)
    age = st.number_input("Property Age (years)", min_value=0, step=1)

    # Submit button
    submitted = st.form_submit_button("Predict Price")


if submitted:
    if rf_model is None:
        st.error(
            "Model is not loaded. Add models/RFmodel.pkl or upload one in the sidebar."
        )
        st.stop()

    prediction_input = pd.DataFrame(
        [[property_tax, insurance, sqft, lot_size, age]],
        columns=FEATURE_COLUMNS,
    )

    try:
        predicted_price = float(rf_model.predict(prediction_input)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Prediction Result")
    st.success(f"Estimated Price: ${predicted_price:,.2f}")

    if reference_df is not None:
        available_features = [
            col for col in FEATURE_COLUMNS if col in reference_df.columns
        ]
        if len(available_features) == len(FEATURE_COLUMNS):
            median_values = reference_df[available_features].median()
            compare_df = pd.DataFrame(
                {
                    "Your Input": prediction_input.iloc[0],
                    "Dataset Median": median_values,
                }
            )
            st.subheader("Your Inputs vs Dataset Median")
            st.bar_chart(compare_df)

st.write(
    """This estimate comes from your trained Random Forest regressor using
property tax, insurance, square footage, lot size, and property age."""
)

st.subheader("Feature Importance")
if rf_model is not None and hasattr(rf_model, "feature_importances_"):
    importances = pd.Series(
        rf_model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    st.bar_chart(importances)

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
elif FEATURE_IMPORTANCE_IMAGE.exists():
    st.image(str(FEATURE_IMPORTANCE_IMAGE), caption="Feature Importance")
else:
    st.info("Feature importance is not available for the loaded model.")

st.subheader("Model Evaluation Matrix")
if reference_df is None:
    st.info("Evaluation matrix is unavailable because no dataset was found.")
else:
    try:
        metrics_df, metrics_note = build_evaluation_matrix(reference_df, rf_model)
        if metrics_df is None:
            st.warning(metrics_note)
        else:
            display_df = metrics_df.copy()
            display_df["MAE"] = display_df["MAE"].map(lambda v: f"{v:,.2f}")
            display_df["RMSE"] = display_df["RMSE"].map(lambda v: f"{v:,.2f}")
            display_df["R2"] = display_df["R2"].map(lambda v: f"{v:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption(metrics_note)
    except Exception as exc:
        st.error(f"Could not compute model evaluation matrix: {exc}")

st.subheader("Market Insights")
if reference_df is None or "price" not in reference_df.columns:
    st.info("Dataset charts are unavailable because no local data file was found.")
else:
    plot_df = reference_df.copy()

    # Additional chart 1: price distribution histogram.
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(plot_df["price"].dropna(), bins=30)
    ax_hist.set_title("Price Distribution")
    ax_hist.set_xlabel("Price")
    ax_hist.set_ylabel("Count")
    st.pyplot(fig_hist)

    # Additional chart 2: relationship between square footage and price.
    if "sqft" in plot_df.columns:
        scatter_data = (
            plot_df[["sqft", "price"]]
            .dropna()
            .sample(n=min(1200, len(plot_df)), random_state=42)
        )
        st.scatter_chart(scatter_data, x="sqft", y="price")
