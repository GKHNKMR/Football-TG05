# app.py
# Streamlit UI for "Over 0.5 – 6 Lig" predictions
# - Shows weekly P(Over 0.5) for EPL, Championship, Serie A, Bundesliga, La Liga, Primeira Liga
# - Works with a Python function `run_week_predictions(...)` in over05_prediction.py
#   OR falls back to reading a local predictions.json file.

import os
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

# ---------- Config ----------
APP_TITLE = "Over 0.5 – 6 Lig Tahminleri"
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


def iso_now_date():
    return datetime.now(timezone.utc).date().isoformat()


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
    # Accept alternative keys and rename to standard
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
    leagues: list[str] | None,
    date_from_utc: str | None,
    date_to_utc: str | None,
) -> pd.DataFrame:
    """
    Loads predictions either by calling user's function or by reading predictions.json.
    Caches result in Streamlit to avoid recomputation during the session.
    """
    # 1) Try user's function
    if callable(run_week_predictions):
        try:
            data = run_week_predictions(
                leagues=leagues,
                date_from_utc=date_from_utc,
                date_to_utc=date_to_utc,
            )
            return to_dataframe(data)
        except Exception as e:
            st.warning(f"run_week_predictions çağrısı başarısız oldu: {e}")

    # 2) Fallback: read predictions.json (should be created by your offline pipeline)
    fallback_path = os.path.join(os.getcwd(), "predictions.json")
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = to_dataframe(data)
            # Optional: filter leagues and dates if columns exist
            if leagues:
                df = df[df["league"].isin(leagues)]
            if "kickoff_utc" in df.columns:
                df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
                if date_from_utc:
                    df = df[df["kickoff_utc"] >= pd.to_datetime(date_from_utc)]
                if date_to_utc:
                    # inclusive end-of-day filter
                    df = df[df["kickoff_utc"] <= pd.to_datetime(date_to_utc) + pd.Timedelta(days=1)]
            return df
        except Exception as e:
            st.error(f"predictions.json okunamadı: {e}")

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
    # Nice formatting
    if "kickoff_utc" in work.columns:
        work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    work["P(>0.5)"] = work["p_over_0_5"].apply(format_percentage)
    show_cols = ["league", "kickoff_utc", "home", "away", "P(>0.5)", "label"]
    show_cols = [c for c in show_cols if c in work.columns]
    return work[show_cols]


def to_download_json(df: pd.DataFrame) -> str:
    # Convert visible subset to a clean JSON list of records
    if df.empty:
        return "[]"
    # Try to reconstruct probabilities from formatted column if necessary
    out = df.copy()
    if "P(>0.5)" in out.columns and "p_over_0_5" not in out.columns:
        out["p_over_0_5"] = out["P(>0.5)"].str.rstrip("%").str.replace(",", "", regex=False).astype(float) / 100.0
    # Rename columns to API-friendly keys
    rename = {
        "P(>0.5)": "p_over_0_5_fmt",
    }
    out = out.rename(columns=rename)
    return json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2)


# ---------- UI ----------
st.set_page_config(page_title="Over 0.5 Radar", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Filtreler")
    selected_leagues = st.multiselect("Lig seç (boş = hepsi)", SUPPORTED_LEAGUES, default=SUPPORTED_LEAGUES)

    start_default, end_default = date_range_default()
    date_from = st.date_input("Başlangıç (UTC)", value=start_default)
    date_to = st.date_input("Bitiş (UTC)", value=end_default)

    min_prob = st.slider("Minimum olasılık eşiği P(>0.5)", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    top_n = st.slider("Gösterilece_

