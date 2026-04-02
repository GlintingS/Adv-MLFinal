from __future__ import annotations

import logging
from pathlib import Path
import warnings

import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

from scr.Model.predict_models import evaluate_model, predict_admission
from scr.Model.train_models import load_model, save_model, train_mlp_model
from scr.data.make_dataset import FEATURE_COLUMNS, load_prepare_split

warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-36s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_data_path() -> Path:
    """Locate the dataset in the current project layout, with legacy fallback."""
    preferred = PROJECT_ROOT / "data" / "raw" / "Admission.csv"
    if preferred.exists():
        return preferred
    fallback = PROJECT_ROOT / "Admission.csv"
    if fallback.exists():
        return fallback
    logger.error("Dataset not found in expected locations")
    raise FileNotFoundError("Dataset not found. Expected data/raw/Admission.csv")


DATA_PATH = resolve_data_path()


def threshold_to_model_path(admit_threshold: float) -> Path:
    """Build a model filename tied to the selected admission threshold."""
    threshold_tag = f"{int(round(admit_threshold * 100)):02d}"
    return PROJECT_ROOT / "models" / f"admission_mlp_thr_{threshold_tag}.pkl"


def compute_attribute_importance(
    model, X_eval: pd.DataFrame, y_eval: pd.Series
) -> pd.DataFrame:
    """Estimate feature importance from validation data using permutation importance."""
    result = permutation_importance(
        model,
        X_eval,
        y_eval,
        n_repeats=15,
        random_state=123,
        scoring="accuracy",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "attribute": X_eval.columns,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)

    total_abs = importance_df["importance"].abs().sum()
    if total_abs > 0:
        importance_df["relative_importance_pct"] = (
            importance_df["importance"].abs() / total_abs * 100
        )
    else:
        importance_df["relative_importance_pct"] = 0.0

    return importance_df.reset_index(drop=True)


@st.cache_resource
def get_or_train_model(
    admit_threshold: float,
    _force_retrain: bool = False,
) -> tuple[object, dict, pd.DataFrame, Path]:
    logger.info(
        "get_or_train_model called with threshold=%.2f, force_retrain=%s",
        admit_threshold,
        _force_retrain,
    )
    try:
        X_train, X_test, y_train, y_test, _ = load_prepare_split(
            DATA_PATH,
            admit_threshold=admit_threshold,
        )
    except Exception:
        logger.exception("Failed to load and prepare data")
        raise

    model_path = threshold_to_model_path(admit_threshold)

    if model_path.exists() and not _force_retrain:
        try:
            model = load_model(model_path)
            logger.info("Loaded existing model from %s", model_path)
        except Exception:
            logger.warning(
                "Failed to load model from %s, retraining", model_path, exc_info=True
            )
            model = train_mlp_model(X_train, y_train)
            save_model(model, model_path)
    else:
        model = train_mlp_model(X_train, y_train)
        save_model(model, model_path)

    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    importance_df = compute_attribute_importance(model, X_test, y_test)
    return model, metrics, importance_df, model_path


st.set_page_config(page_title="UCLA Admission Predictor", page_icon="🎓", layout="wide")

st.sidebar.header("Model Controls")
admit_threshold = st.sidebar.slider(
    "Admission Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.01,
    help="Rows with Admit_Chance >= threshold become class 1 during training.",
)

retrain_clicked = st.sidebar.button("Retrain Model With Current Threshold")
if retrain_clicked:
    st.session_state.force_retrain = True
else:
    st.session_state.force_retrain = False

st.title("UCLA Admission Chance Classifier")
st.write(
    "Predict whether a student is likely to be admitted based on profile features "
    f"from `{DATA_PATH.relative_to(PROJECT_ROOT)}` (current threshold: {admit_threshold:.2f})."
)

try:
    model, metrics, importance_df, model_path = get_or_train_model(
        admit_threshold,
        st.session_state.get("force_retrain", False),
    )
except Exception as exc:
    st.error(f"Failed to load or train model: {exc}")
    logger.exception("Model loading/training error")
    st.stop()
st.caption(f"Model artifact: `{model_path.name}`")

col_a, col_b = st.columns([2, 1])

with col_a:
    with st.form("admission_form"):
        gre_score = st.slider("GRE Score", min_value=260, max_value=340, value=320)
        toefl_score = st.slider("TOEFL Score", min_value=0, max_value=120, value=108)
        university_rating = st.selectbox(
            "University Rating", options=[1, 2, 3, 4, 5], index=2
        )
        sop = st.slider(
            "SOP Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5
        )
        lor = st.slider(
            "LOR Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5
        )
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
        research = st.selectbox("Research Experience", options=[0, 1], index=1)

        submitted = st.form_submit_button("Predict Admission")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "GRE_Score": gre_score,
                    "TOEFL_Score": toefl_score,
                    "University_Rating": str(university_rating),
                    "SOP": sop,
                    "LOR": lor,
                    "CGPA": cgpa,
                    "Research": str(research),
                }
            ]
        )
        row = row[FEATURE_COLUMNS]

        try:
            pred_class, pred_prob = predict_admission(model, row)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            logger.exception("Prediction error")
            pred_class, pred_prob = None, None

        if pred_class is not None:
            st.subheader("Prediction")
            st.metric("Admission Probability", f"{pred_prob:.2%}")
            if pred_class == 1:
                st.success("Likely to be admitted (class = 1)")
            else:
                st.error("Unlikely to be admitted (class = 0)")

with col_b:
    st.subheader("Model Performance")
    st.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
    st.caption("Model: MLPClassifier with preprocessing pipeline")

st.divider()
st.subheader("Attribute Importance")
st.caption("Permutation importance on the test split (higher means more influence).")
st.bar_chart(
    importance_df.set_index("attribute")[["importance"]],
    use_container_width=True,
)
st.dataframe(
    importance_df[
        ["attribute", "importance", "importance_std", "relative_importance_pct"]
    ],
    use_container_width=True,
)

st.divider()
st.subheader("Dataset Preview")
st.dataframe(pd.read_csv(DATA_PATH).head(10), use_container_width=True)
