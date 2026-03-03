"""
app.py

Streamlit web application for the Intelligent Crop Yield Prediction system.
Provides crop yield predictions, data exploration, and model performance analysis.

Usage:
    streamlit run app.py
"""

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Intelligent Crop Yield Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #2E7D32;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open("models/crop_yield_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return model, label_encoders, scaler, model_columns, metrics


try:
    model, label_encoders, scaler, model_columns, metrics = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Predict Yield", "Data Explorer", "Model Performance"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Intelligent Crop Yield Prediction")
st.sidebar.caption("ML-Based Agricultural Analytics System")


# PAGE 1: PREDICT YIELD
if page == "Predict Yield":
    st.markdown('<p class="main-header">Crop Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enter agricultural parameters to predict crop yield (hg/ha)</p>',
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.error("Model artifacts not found. Run `python src/train_model.py` first.")
        st.stop()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crop & Region")
        crop_options = list(label_encoders["Item"].classes_)
        selected_crop = st.selectbox("Crop Type", crop_options, index=0)

        area_options = list(label_encoders["Area"].classes_)
        selected_area = st.selectbox("Country / Region", area_options, index=0)

        selected_year = st.number_input(
            "Year", min_value=1960, max_value=2030, value=2024, step=1
        )

    with col2:
        st.subheader("Environmental Factors")
        rainfall = st.number_input(
            "Average Rainfall (mm/year)", min_value=0.0, max_value=5000.0,
            value=1000.0, step=10.0,
        )
        pesticides = st.number_input(
            "Pesticides Used (tonnes)", min_value=0.0, max_value=500000.0,
            value=1000.0, step=100.0,
        )
        avg_temp = st.number_input(
            "Average Temperature (C)", min_value=-10.0, max_value=50.0,
            value=25.0, step=0.5,
        )

    st.divider()

    if st.button("Predict Yield", use_container_width=True):
        crop_encoded = label_encoders["Item"].transform([selected_crop])[0]
        area_encoded = label_encoders["Area"].transform([selected_area])[0]

        input_data = {}
        for col in model_columns:
            if col == "Area":
                input_data[col] = area_encoded
            elif col == "Item":
                input_data[col] = crop_encoded
            elif col == "Year":
                input_data[col] = selected_year
            elif col == "average_rain_fall_mm_per_year":
                input_data[col] = rainfall
            elif col == "pesticides_tonnes":
                input_data[col] = pesticides
            elif col == "avg_temp":
                input_data[col] = avg_temp
            else:
                input_data[col] = 0

        input_df = pd.DataFrame([input_data])

        numeric_cols = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
        cols_to_scale = [c for c in numeric_cols if c in input_df.columns]
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        prediction = model.predict(input_df)[0]

        st.success(f"### Predicted Yield: **{prediction:,.2f} hg/ha**")

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric("Crop", selected_crop)
        with result_col2:
            st.metric("Region", selected_area)
        with result_col3:
            st.metric("Predicted Yield", f"{prediction:,.2f} hg/ha")


# PAGE 2: DATA EXPLORER
elif page == "Data Explorer":
    st.markdown('<p class="main-header">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload and explore crop yield data</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a CSV file (or the built-in dataset will be used)", type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    else:
        try:
            df = pd.read_csv("data/yield_df.csv")
            st.info("Using built-in dataset: data/yield_df.csv")
        except FileNotFoundError:
            st.warning("No dataset found. Upload a CSV file to explore.")
            st.stop()

    st.subheader("Dataset Overview")
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    with overview_col1:
        st.metric("Rows", df.shape[0])
    with overview_col2:
        st.metric("Columns", df.shape[1])
    with overview_col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
    with overview_col4:
        st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distribution Plots")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select a numeric column", numeric_cols)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(df[selected_col].dropna(), bins=30, color="#2E7D32", edgecolor="white", alpha=0.8)
        axes[0].set_title(f"Distribution of {selected_col}")
        axes[0].set_xlabel(selected_col)
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(df[selected_col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#C8E6C9"))
        axes[1].set_title(f"Box Plot of {selected_col}")
        axes[1].set_ylabel(selected_col)

        plt.tight_layout()
        st.pyplot(fig)

    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="Greens", fmt=".2f",
                    ax=ax_corr, linewidths=0.5)
        ax_corr.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig_corr)


# PAGE 3: MODEL PERFORMANCE
elif page == "Model Performance":
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Evaluation metrics and feature importance analysis</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    if not model_loaded:
        st.error("Model artifacts not found. Run `python src/train_model.py` first.")
        st.stop()

    best_model_name = metrics.get("best_model", "N/A")
    st.success(f"**Best Model: {best_model_name}**")

    st.subheader("Model Comparison")

    model_names = [k for k in metrics.keys() if k not in ("best_model", "feature_importance")]
    comparison_data = []
    for name in model_names:
        m = metrics[name]
        comparison_data.append({
            "Model": name,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2 Score": m["R2"],
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.subheader("R2 Score Comparison")
    fig_r2, ax_r2 = plt.subplots(figsize=(8, 4))
    colors = ["#81C784" if name != best_model_name else "#2E7D32" for name in comparison_df["Model"]]
    bars = ax_r2.barh(comparison_df["Model"], comparison_df["R2 Score"], color=colors, edgecolor="white")
    ax_r2.set_xlabel("R2 Score")
    ax_r2.set_title("Model R2 Score Comparison")
    for bar, val in zip(bars, comparison_df["R2 Score"]):
        ax_r2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                   f"{val:.4f}", va="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_r2)

    feature_importance = metrics.get("feature_importance", {})
    if feature_importance:
        st.subheader("Feature Importance")

        fi_df = pd.DataFrame(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature", "Importance"],
        )

        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        ax_fi.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
                    color="#66BB6A", edgecolor="white")
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_title("Yield-Driving Features")
        plt.tight_layout()
        st.pyplot(fig_fi)

        st.markdown("The features above are the primary drivers of crop yield "
                     "as identified by the trained model. Higher importance indicates a stronger "
                     "influence on the predicted yield value.")
    else:
        st.info("Feature importance data not available.")


st.divider()
st.caption("Intelligent Crop Yield Prediction — ML-Based Agricultural Analytics System")
