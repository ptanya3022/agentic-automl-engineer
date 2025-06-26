import streamlit as st
import pandas as pd
from pipeline import run_pipeline

st.set_page_config(page_title="🤖 Agentic AutoML Engineer", layout="centered")

st.title("🤖 Welcome to Your AI-powered AutoML Engineer!")
st.markdown("Upload any **CSV dataset**, select your **target column**, and let the agent build, tune, and evaluate the best ML model for you.")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV only)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # Dropdown to select target column
        target_column = st.selectbox("🎯 Select Target Column", df.columns)

        # Run AutoML Button
        if st.button("🚀 Run AutoML Pipeline"):
            st.success(f"✅ Running AutoML pipeline with target: `{target_column}`")

            with st.spinner("Training models, optimizing with Optuna... please wait..."):
                best_model, best_accuracy, report_df = run_pipeline(df, target_column)

            st.subheader("📊 Model Performance Summary")
            st.write(f"**Best Model:** `{type(best_model).__name__}`")
            st.write(f"**Best Accuracy:** `{best_accuracy:.2f}`")
            st.write("**Classification Report:**")
            st.dataframe(report_df)

    except Exception as e:
        st.error(f"❌ Something went wrong:\n\n{e}")
