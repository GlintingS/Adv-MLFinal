import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make scr/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "scr"))

from data import make_dataset
from Model import train_models, predict_models
from visuals import visualize

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("Mall Customer Segmentation")
st.write(
    """
Unsupervised K-Means clustering app that segments mall customers based on
their **Annual Income** and **Spending Score** (and optionally **Age**).
"""
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data
def load_data_from_disk():
    return make_dataset.load_data()


@st.cache_data
def load_data_from_upload(file_bytes):
    import io

    if len(file_bytes) == 0:
        return None
    return pd.read_csv(io.BytesIO(file_bytes))


# Try loading from disk first; fall back to an in-app uploader
df = None
try:
    df = load_data_from_disk()
except Exception:
    pass

if df is None:
    st.warning(
        "**`mall_customers.csv` could not be read from disk** "
        "(the file may not be synced from OneDrive yet). "
        "Upload it below to continue."
    )
    uploaded = st.file_uploader("Upload mall_customers.csv", type="csv")
    if uploaded is not None:
        file_bytes = uploaded.read()
        if len(file_bytes) == 0:
            st.error(
                "The file you uploaded is **0 bytes** — it is an unsynced OneDrive "
                "placeholder, not the real file.\n\n"
                "**To get the real file:** right-click `mall_customers.csv` in Finder "
                "→ *Always keep on this device*, wait for it to sync (the cloud icon "
                "disappears), then upload it again."
            )
            st.stop()
        df = load_data_from_upload(file_bytes)
        st.success("File loaded successfully.")
    else:
        st.info(
            "Alternatively, right-click `mall_customers.csv` in Finder → "
            "*Always keep on this device*, then refresh this page."
        )
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

feature_set = st.sidebar.radio(
    "Feature set",
    options=["2D — Income & Spending", "3D — Age, Income & Spending"],
)

if feature_set.startswith("2D"):
    feature_cols = ["Annual_Income", "Spending_Score"]
else:
    feature_cols = ["Age", "Annual_Income", "Spending_Score"]

k_range_start = st.sidebar.slider("K range start", min_value=2, max_value=5, value=3)
k_range_end = st.sidebar.slider("K range end", min_value=5, max_value=12, value=8)
n_clusters = st.sidebar.slider(
    "Final number of clusters (k)", min_value=2, max_value=10, value=5
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_eda, tab_elbow, tab_sil, tab_clusters = st.tabs(
    ["📊 Data Overview", "📈 Elbow Method", "📉 Silhouette Method", "🔵 Clusters"]
)

X = make_dataset.get_features(df, feature_cols)
k_range = range(k_range_start, k_range_end + 1)

# ── EDA tab ─────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Raw Data")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.subheader("Correlation Matrix")
        st.dataframe(
            df.corr(numeric_only=True).style.background_gradient(cmap="coolwarm"),
            use_container_width=True,
        )

    st.subheader("Pair Plot")
    fig_pair = visualize.plot_pairplot(df, ["Age", "Annual_Income", "Spending_Score"])
    st.pyplot(fig_pair)

# ── Elbow tab ────────────────────────────────────────────────────────────────
with tab_elbow:
    st.subheader("Elbow Plot (WCSS)")
    st.write("The 'elbow' in the curve suggests the optimal number of clusters.")

    with st.spinner("Computing WCSS scores…"):
        wss_df = train_models.run_elbow_method(X, k_range=k_range)

    st.dataframe(wss_df, use_container_width=True)
    fig_elbow = visualize.plot_elbow(wss_df)
    st.pyplot(fig_elbow)

# ── Silhouette tab ───────────────────────────────────────────────────────────
with tab_sil:
    st.subheader("Silhouette Plot")
    st.write("Silhouette score closer to **+1** means better-defined clusters.")

    with st.spinner("Computing Silhouette scores…"):
        sil_df = train_models.run_silhouette(X, k_range=k_range)

    st.dataframe(sil_df, use_container_width=True)
    fig_sil = visualize.plot_silhouette(sil_df)
    st.pyplot(fig_sil)

    best_k = int(sil_df.loc[sil_df["Silhouette_Score"].idxmax(), "cluster"])
    st.success(f"Best k by Silhouette Score: **{best_k}**")

# ── Clusters tab ─────────────────────────────────────────────────────────────
with tab_clusters:
    st.subheader(f"K-Means Clustering  (k = {n_clusters})")

    with st.spinner("Training K-Means model…"):
        kmodel = train_models.train_kmeans(X, n_clusters=n_clusters)

    df_clustered = df.copy()
    df_clustered["Cluster"] = predict_models.predict_clusters(kmodel, X)

    st.write("**Cluster distribution**")
    st.dataframe(
        df_clustered["Cluster"]
        .value_counts()
        .rename("Count")
        .reset_index()
        .rename(columns={"index": "Cluster"}),
        use_container_width=True,
    )

    if feature_set.startswith("2D"):
        centers = kmodel.cluster_centers_
        fig_scatter = visualize.plot_clusters(
            df_clustered, "Annual_Income", "Spending_Score", "Cluster", centers
        )
        st.pyplot(fig_scatter)
    else:
        st.info(
            "Scatter plot is only available for the 2-feature model. "
            "Switch to **2D** in the sidebar to visualise clusters."
        )

    st.subheader("Clustered Data Sample")
    st.dataframe(df_clustered.sample(10, random_state=42), use_container_width=True)
