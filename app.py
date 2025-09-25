# app.py (FINAL – robust fallbacks, uploader, embedded seed)
# - Tries API (run_week_predictions) first
# - Then tries predictions.json from multiple locations; skips empty JSONs
# - Supports explicit path via PREDICTIONS_JSON_PATH
# - Allows user to upload JSON/CSV from the sidebar
# - If everything fails, uses embedded SEED_PREDICTIONS (so the UI never stays empty)

import os
import json
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Over 0.5 – 6 League Predictions", layout="wide")

# ---------- Thresholds & leagues ----------
ULTRA_TH = 0.98
HIGH_TH = 0.95
SUPPORTED_LEAGUES = [
    "Premier League", "Championship", "Serie A", "Bundesliga", "La Liga", "Primeira Liga"
]

# ---------- Embedded seed predictions (as a last resort) ----------
SEED_PREDICTIONS = [
    {"match_id":"PL20250926-001","league":"Premier League","kickoff_utc":"2025-09-26T19:00:00Z","home":"Arsenal","away":"Chelsea","p_over_0_5":0.981,"label":"ULTRA"},
    {"match_id":"PL20250927-002","league":"Premier League","kickoff_utc":"2025-09-27T16:30:00Z","home":"Liverpool","away":"Manchester United","p_over_0_5":0.965,"label":"HIGH"},
    {"match_id":"CH20250929-003","league":"Championship","kickoff_utc":"2025-09-29T19:00:00Z","home":"Leeds United","away":"Sunderland","p_over_0_5":0.942,"label":""},
    {"match_id":"SA20250928-004","league":"Serie A","kickoff_utc":"2025-09-28T18:45:00Z","home":"AC Milan","away":"Inter Milan","p_over_0_5":0.989,"label":"ULTRA"},
    {"match_id":"SA20250930-005","league":"Serie A","kickoff_utc":"2025-09-30T17:30:00Z","home":"Juventus","away":"Roma","p_over_0_5":0.958,"label":"HIGH"},
    {"match_id":"BL20250927-006","league":"Bundesliga","kickoff_utc":"2025-09-27T14:30:00Z","home":"Bayern Munich","away":"Borussia Dortmund","p_over_0_5":0.993,"label":"ULTRA"},
    {"match_id":"BL20250927-007","league":"Bundesliga","kickoff_utc":"2025-09-27T16:30:00Z","home":"RB Leipzig","away":"Union Berlin","p_over_0_5":0.948,"label":"HIGH"},
    {"match_id":"LL20250928-008","league":"La Liga","kickoff_utc":"2025-09-28T20:00:00Z","home":"Barcelona","away":"Real Madrid","p_over_0_5":0.991,"label":"ULTRA"},
    {"match_id":"P120250930-009","league":"Primeira Liga","kickoff_utc":"2025-09-30T20:15:00Z","home":"Benfica","away":"Porto","p_over_0_5":0.977,"label":"ULTRA"},
    {"match_id":"LL20251001-010","league":"La Liga","kickoff_utc":"2025-10-01T19:00:00Z","home":"Atletico Madrid","away":"Sevilla","p_over_0_5":0.954,"label":"HIGH"}
]

# ---------- Try importing API function ----------
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
    today = datetime.now(timezone.utc).date()
    return today, today + timedelta(days=7)


def to_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data or [])

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

    for col in ["league", "home", "away", "p_over_0_5"]:
        if col not in df.columns:
            df[col] = None

    if "label" not in df.columns:
        df["label"] = df["p_over_0_5"].apply(lambda p: add_label(float(p)) if pd.notnull(p) else "")

    ordered = ["league", "kickoff_utc", "home", "away", "p_over_0_5", "label", "match_id"]
    left = [c for c in ordered if c in df.columns]
    right = [c for c in df.columns if c not in left]
    return df[left + right]


def candidate_json_paths() -> list[Path]:
    paths = []
    env_path = os.getenv("PREDICTIONS_JSON_PATH")
    if env_path:
        paths.append(Path(env_path).resolve())

    cwd = Path(os.getcwd()).resolve()
    here = Path(__file__).resolve().parent
    paths.extend([
        (cwd / "predictions.json").resolve(),
        (here / "predictions.json").resolve(),
        (here.parent / "predictions.json").resolve(),
    ])

    uniq = []
    for p in paths:
        if p not in uniq:
            uniq.append(p)
    return uniq


def load_local_json(leagues: Optional[List[str]], date_from_utc: Optional[str],
                    date_to_utc: Optional[str], ignore_date_filter: bool) -> pd.DataFrame:
    last_error = None
    tried = []

    for p in candidate_json_paths():
        tried.append(str(p))
        if p.exists():
            try:
                raw = p.read_text(encoding="utf-8").strip()
                if not raw:
                    st.info(f"Found empty file (0 bytes): {p}")
                    continue

                data = json.loads(raw)
                df = to_dataframe(data)
                total_rows = len(df)
                st.info(f"Loaded {total_rows} rows from: {p}")
                if total_rows == 0:
                    continue

                before_league = len(df)
                if leagues:
                    df = df[df["league"].isin(leagues)]
                st.info(f"League filter kept {len(df)} of {before_league} rows.")

                parse_ok = False
                if "kickoff_utc" in df.columns:
                    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
                    valid_dt = df["kickoff_utc"].notna().sum()
                    parse_ok = valid_dt > 0
                    st.info(f"'kickoff_utc' parsed valid timestamps: {valid_dt}/{len(df)}")

                if not ignore_date_filter and parse_ok:
                    before_date = len(df)
                    if date_from_utc:
                        df = df[df["kickoff_utc"] >= pd.to_datetime(date_from_utc, utc=True, errors="coerce")]
                    if date_to_utc:
                        df = df[df["kickoff_utc"] <= pd.to_datetime(date_to_utc, utc=True, errors="coerce") + pd.Timedelta(days=1)]
                    st.info(f"Date filter kept {len(df)} of {before_date} rows.")
                else:
                    if ignore_date_filter:
                        st.info(f"Ignoring date filter (showing {len(df)} rows).")
                    elif not parse_ok:
                        st.info(f"'kickoff_utc' parsing failed; showing {len(df)} rows without date filtering.")
                return df
            except Exception as e:
                last_error = e
                st.warning(f"Failed to read {p}: {e}")
                continue

    if last_error:
        st.error(f"Could not read predictions.json. Last error: {last_error}")
    st.warning("Checked these locations but found no usable predictions.json:\n" + "\n".join(f"- {p}" for p in tried))
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_predictions_cached(leagues: Optional[List[str]], date_from_utc: Optional[str],
                            date_to_utc: Optional[str], ignore_date_filter: bool,
                            refresh_key: Optional[str] = None) -> pd.DataFrame:
    # 1) Try API
    if callable(run_week_predictions):
        try:
            data = run_week_predictions(leagues=leagues, date_from_utc=date_from_utc, date_to_utc=date_to_utc)
            df = to_dataframe(data)
            if not df.empty:
                return df
            else:
                st.info("API returned no rows; falling back to local predictions.json.")
        except Exception as e:
            st.warning(f"run_week_predictions failed ({e}); falling back to local predictions.json.")

    # 2) Local JSON fallback
    return load_local_json(leagues, date_from_utc, date_to_utc, ignore_date_filter)


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
    work["p_over_0_5"] = pd.to_numeric(work["p_over_0_5"], errors="coerce")
    work = work.dropna(subset=["p_over_0_5"])
    work = work[work["p_over_0_5"] >= min_prob].sort_values("p_over_0_5", ascending=False)
    if top_n > 0:
        work = work.head(top_n)
    if "kickoff_utc" in work.columns:
        work["kickoff_utc"] = pd.to_datetime(work["kickoff_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    work["P(>0.5)"] = work["p_over_0_5"].apply(format_percentage)
    cols = [c for c in ["league", "kickoff_utc", "home", "away", "P(>0.5)", "label"] if c in work.columns]
    return work[cols]


def to_download_json(df: pd.DataFrame) -> str:
    if df.empty:
        return "[]"
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)


# ---------- UI ----------
st.title("Over 0.5 – 6 League Predictions")

with st.sidebar:
    st.subheader("Filters")
    selected_leagues = st.multiselect("Select League (empty = all)", SUPPORTED_LEAGUES, default=SUPPORTED_LEAGUES)

    start_default, end_default = date_range_default()
    date_from = st.date_input("Start Date (UTC)", value=start_default)
    date_to = st.date_input("End Date (UTC)", value=end_default)

    ignore_date_filter = st.checkbox("Ignore date filter", value=False)

    st.markdown("---")
    uploaded = st.file_uploader("Upload predictions file (JSON or CSV)", type=["json", "csv"])

    st.markdown("---")
    min_prob = st.slider("Minimum probability threshold P(>0.5)", 0.50, 0.99, 0.95, 0.01)
    top_n = st.slider("Top N matches to display", 10, 200, 50, 10)
    st.caption("Label thresholds: ULTRA ≥ 0.98, HIGH ≥ 0.95")

# Refresh button
if st.button("Fetch / Refresh Predictions", type="primary"):
    st.session_state["refresh_ts"] = datetime.now().isoformat()

refresh_key = st.session_state.get("refresh_ts", "init")

# 1) Try API / Local JSON cache
with st.spinner("Loading predictions..."):
    df_all = load_predictions_cached(
        leagues=selected_leagues if selected_leagues else None,
        date_from_utc=str(date_from),
        date_to_utc=str(date_to),
        ignore_date_filter=ignore_date_filter,
        refresh_key=refresh_key,
    )

# 2) If empty and user uploaded a file, parse it
if df_all.empty and uploaded is not None:
    try:
        if uploaded.type.endswith("json"):
            data = json.load(uploaded)
            df_all = to_dataframe(data)
            st.success(f"Using uploaded JSON file. Rows: {len(df_all)}")
        else:
            text = uploaded.getvalue().decode("utf-8", errors="ignore")
            df_csv = pd.read_csv(StringIO(text))
            df_all = to_dataframe(df_csv)
            st.success(f"Using uploaded CSV file. Rows: {len(df_all)}")
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")

# 3) If still empty, use embedded seed
if df_all.empty:
    st.info("Falling back to embedded demo data (SEED_PREDICTIONS).")
    df_all = to_dataframe(SEED_PREDICTIONS)

# ---- Show table ----
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

st.download_button(
    label="Download Filtered Results as JSON",
    data=to_download_json(df_view).encode("utf-8"),
    file_name="predictions_filtered.json",
    mime="application/json",
)

st.caption("This app provides statistical predictions only; it is not betting advice. Users are responsible for following local laws.")
