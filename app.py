import streamlit as st
import pandas as pd

INPUT_FILE = "predictions.xlsx"
FIXTURE_SHEET = "Fixture"

st.set_page_config(page_title="Over 0.5 Predictions", layout="wide")

st.title("⚽ Over 0.5 Goal Predictions")

# Load predictions
try:
    df = pd.read_excel(INPUT_FILE, sheet_name=FIXTURE_SHEET)
except Exception as e:
    st.error(f"Could not load predictions: {e}")
    st.stop()

# Filters
leagues = df["League"].unique() if "League" in df.columns else []
selected_leagues = st.multiselect("Select League(s)", leagues, default=list(leagues))

if selected_leagues:
    df = df[df["League"].isin(selected_leagues)]

# Show table
st.dataframe(df, use_container_width=True)

# Download
st.download_button(
    "Download Predictions (Excel)",
    data=open(INPUT_FILE, "rb").read(),
    file_name="predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Probabilities are estimated with a simple Poisson model from past matches.")
