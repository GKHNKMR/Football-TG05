# app.py
# Streamlit UI for "Over 0.5 – 6 League Predictions"
# - Shows weekly P(Over 0.5) for EPL, Championship, Serie A, Bundesliga, La Liga, Primeira Liga
# - Calls `run_week_predictions(...)` in over05_prediction.py
# - Falls back to reading a local predictions.json when API returns empty or fails

import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Over 0.5 Radar", layout="wide")

# ---------- Constants ----------
APP_TITLE = "Over 0.5 – 6 League Predictions"
ULTRA_TH = 0.98
HIGH_TH = 0.95

SUPPORTED_LEAGUES = [
    "Premier League",
    "Championship",
    "Serie A",
    "Bundesliga",
    "La Liga",
    "Primeira Liga",
]

# ---------- Optional import of your prediction function ----------
# Expected signature:
#   run_week_predictions(leagues: list[str] | None, date_from_utc: str | None, date_to_utc: str | None) -> list[dict] | pd.DataFrame
# It should return records with at least: league, home, away, p_over_0_5, kickoff_utc (ISO)
run_week_predictions = None
try:
    from over05_prediction import run_week_predictions as _run
    run_week_predictions = _run
except Exception:
    run_week_predictions = None


# ---------- Helpers ----------
def add_label(p: float) -> str:
    if p is None:
        return ""
    if p >= ULTRA_TH:
        return "ULTRA"
    if p >= HIGH_TH:
        return "HIGH"
    return ""


def date_range_default():
    # Default to the next 7 days (UTC)
    today = datetime.now(timezone.utc).date()
    return today, today + timedelta(days=7)


def to_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data or [])
    # Normalize expected columns
    rename_map = {
        "fixture_id": "match_id",
        "matchId": "match_id",
        "homeTeam": "home",
        "awayTeam": "away",
        "league_name": "league",
        "kickoff": "kickoff_utc",
        "kickoffUTC": "kickoff_utc",
        "pOver0_5": "p_over_0_5",
        "p_over0_5": "p_over_0_5",
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    for col in ["league", "home", "away", "p_over_0_5"]:
        if col not in df.columns:
            df[col] = None

    # Label column
    if "label" not in df.columns:
        df["label"] = df["p_over_0_5"].apply(lambda p: add_label(float(p)) if pd.notnull(p) else "")

    # Order columns
    ordered_cols = ["league", "kickoff_utc", "home", "away", "p_over_0_5", "label", "match_id"]
    left = [c for c in ordered_cols if c in df.columns]
    right = [c for c in df.columns if c not in left]
    df = df[left + right]
    return df


@st.cache_data(show_spinner=False)
def load_predictions_cached(
    leagues: Optional[List[str]],
    date_from_utc: Optional[str],
    date_to_utc: Optional[str],
    refresh_key: Optional[str] = None,  # cache bust key
) -> pd.DataFrame:
    """
    Loads predictions by calling user's function.
    If the result is empty or the call fails, it falls back to predictions.json.
    """
    # 1) Try user's function first
    if callable(run_week_predictions):
        try:
            data = run_week_predictions(
                leagues=leagues,
                date_from_utc=date_from_utc,
                date_to_utc=date_to_utc,
            )
            df = to_dataframe(data)
            if not df.empty:
                return df
            else:
                st.info("API returned no rows; falling back to local predictions.json.")
        except Exception as e:
            st.warning(f"run_week_predictions call failed ({e}); falling back to local predictions.json.")

    # 2) Fallback: read predictions.json (created by your offline pipeline or Actions)
    fallback_path = os.path.join(os.getcwd(), "predictions.json")
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = to_dataframe(data)
            # Optional filters
            if leagues:
                df = df[df["league"].isin(leagues)]
            if "kickoff_utc" in df.columns:
                df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
                if date_from_utc:
                    df = df[df["kickoff_utc"] >= pd.to_datetime(date_from_utc)]
                if date_to_utc:
                    df = df[df["kickoff_utc"] <= pd.to_datetime(date_to_utc) + pd.Timedelta(days=1)]
            return df
        except Exception as e:
            st.error(f"Could not read predictions.json: {e}")

    # Nothing worked
    return pd.DataFrame()


def format_percentage(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    try:
        return f"{float(p)*100:,.2f}%"
    except Exception:
        return ""


def filtered_view(df: pd.DataFrame, min_prob: float, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    # Ensure numeric
    work["p_over_0_5"] = pd.to_numeric(work["p_over_0_5"], errors="coerce")
    work = work.dropna(subset=["p_over_0_5"])
    # Filter and sort
    work = work[work["p_over_0_5"] >= min_prob].sort_values("p_over_0_5", ascending=False)
    # Top N
    if top_n > 0:
        work = work.head(top_n)
    # Pretty formatting
    if "kickoff_utc" in work.columns:
        work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    work["P(>0.5)"] = work["p_over_0_5"].apply(format_percentage)
    show_cols = ["league", "kickoff_utc", "home", "away", "P(>0.5)", "label"]
    show_cols = [c for c in show_cols if c in work.columns]
    return work[show_cols]


def to_download_json(df: pd.DataFrame) -> str:
    if df.empty:
        return "[]"
    out = df.copy()
    if "P(>0.5)" in out.columns and "p_over_0_5" not in out.columns:
        out["p_over_0_5"] = out["P(>0.5)"].str.rstrip("%").str.replace(",", "", regex=False).astype(float) / 100.0
    rename = {"P(>0.5)": "p_over_0_5_fmt"}
    out = out.rename(columns=rename)
    return json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2)


# ---------- UI ----------
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Filters")
    selected_leagues = st.multiselect("Select League (empty = all)", SUPPORTED_LEAGUES, default=SUPPORTED_LEAGUES)

    start_default, end_default = date_range_default()
    date_from = st.date_input("Start Date (UTC)", value=start_default)
    date_to = st.date_input("End Date (UTC)", value=end_default)

    min_prob = st.slider("Minimum probability threshold P(>0.5)", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    top_n = st.slider("Top N matches to display", min_value=10, max_value=200, value=50, step=10)

    st.caption("Label thresholds: ULTRA ≥ 0.98, HIGH ≥ 0.95")

col_left, col_right = st.columns([3, 2])

with col_left:
    if st.button("Fetch / Refresh Predictions", type="primary"):
        st.session_state["refresh_ts"] = datetime.now().isoformat()

# cache-busting key updated on button click
refresh_key = st.session_state.get("refresh_ts", "init")

with st.spinner("Loading predictions..."):
    df_all = load_predictions_cached(
        leagues=selected_leagues if selected_leagues else None,
        date_from_utc=str(date_from),
        date_to_utc=str(date_to),
        refresh_key=refresh_key,  # IMPORTANT: cache bust when user refreshes
    )

if df_all.empty:
    st.warning("No predictions found. (API returned nothing and predictions.json fallback is missing or invalid.)")
else:
    df_view = filtered_view(df_all, min_prob=min_prob, top_n=top_n)
    st.subheader("Results")
    st.dataframe(df_view, use_container_width=True, height=600)

    with st.expander("Summary / Stats"):
        total = len(df_all)
        shown = len(df_view)
        ultra_cnt = (df_view["label"] == "ULTRA").sum() if "label" in df_view.columns else 0
        high_cnt = (df_view["label"] == "HIGH").sum() if "label" in df_view.columns else 0
        st.markdown(
            f"- Total matches (before filters): **{total}**\n"
            f"- Displayed matches: **{shown}**\n"
            f"- ULTRA (≥98%): **{ultra_cnt}** | HIGH (≥95%): **{high_cnt}**"
        )

    # Download filtered JSON
    json_payload = to_download_json(df_view)
    st.download_button(
        label="Download Filtered Results as JSON",
        data=json_payload.encode("utf-8"),
        file_name="predictions_filtered.json",
        mime="application/json",
    )

st.caption(" This app provides statistical predictions only; it is not betting advice. Users are responsible for following local laws.")
