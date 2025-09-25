"""Generate simulated football fixture probabilities for over 0.5 total goals.

This module mirrors the workflow described in the user prompt:

1. Simulate historical results for a set of leagues.
2. Engineer several match features (head-to-head, team form, etc.).
3. Train and calibrate a Gradient Boosting model.
4. Predict the probability of ``Total Goals Over 0.5`` for upcoming fixtures.
5. Pretty-print the resulting table of predictions.

Running the script will output a ranked list of the simulated fixtures
along with their predicted probability of finishing with at least one goal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import Dict, Iterable, List

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier


@dataclass(frozen=True)
class LeagueConfig:
    """Configuration of the leagues for which we simulate data."""

    name: str
    historical_matches: int = 1000
    upcoming_fixtures: int = 5


def create_historical_data(leagues: Iterable[LeagueConfig]) -> pd.DataFrame:
    """Simulate historical match data for roughly the last 5.5 years."""

    data: List[List[object]] = []
    end_date = datetime(2025, 9, 25)
    start_date = end_date - timedelta(days=int(5 * 365 + 365 / 2))

    for league in leagues:
        for _ in range(league.historical_matches):
            match_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            home_team = f"{league.name} Home Team {random.randint(1, 20)}"
            away_team = f"{league.name} Away Team {random.randint(1, 20)}"
            home_goals = random.randint(0, 5)
            away_goals = random.randint(0, 5)
            data.append(
                [match_date, league.name, home_team, away_team, home_goals, away_goals]
            )

    df = pd.DataFrame(
        data,
        columns=[
            "match_date",
            "league",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ],
    )
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def create_current_fixtures(leagues: Iterable[LeagueConfig]) -> pd.DataFrame:
    """Simulate a small slate of upcoming fixtures."""

    data: List[List[object]] = []
    fixture_date = datetime(2025, 9, 28)

    for league in leagues:
        for _ in range(league.upcoming_fixtures):
            home_team = f"{league.name} Home Team {random.randint(1, 20)}"
            away_team = f"{league.name} Away Team {random.randint(1, 20)}"
            data.append([fixture_date, league.name, home_team, away_team])

    df = pd.DataFrame(data, columns=["match_date", "league", "home_team", "away_team"])
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def calculate_h2h_stats(
    df: pd.DataFrame, team1: str, team2: str, end_date: datetime, years: int = 5
) -> Dict[str, int]:
    """Calculate head-to-head summary statistics."""

    start_date = end_date - timedelta(days=years * 365)
    h2h_matches = df[
        ((df["home_team"] == team1) & (df["away_team"] == team2))
        | ((df["home_team"] == team2) & (df["away_team"] == team1))
    ]
    recent_h2h_matches = h2h_matches[h2h_matches["match_date"] >= start_date]

    if recent_h2h_matches.empty:
        return {
            "h2h_wins": 0,
            "h2h_losses": 0,
            "h2h_draws": 0,
            "h2h_goals_scored": 0,
            "h2h_goals_conceded": 0,
            "h2h_total_matches": 0,
        }

    team1_wins = 0
    team2_wins = 0
    draws = 0
    team1_goals_scored = 0
    team1_goals_conceded = 0

    for _, row in recent_h2h_matches.iterrows():
        if row["home_team"] == team1 and row["away_team"] == team2:
            if row["home_goals"] > row["away_goals"]:
                team1_wins += 1
            elif row["home_goals"] < row["away_goals"]:
                team2_wins += 1
            else:
                draws += 1
            team1_goals_scored += row["home_goals"]
            team1_goals_conceded += row["away_goals"]
        elif row["home_team"] == team2 and row["away_team"] == team1:
            if row["away_goals"] > row["home_goals"]:
                team1_wins += 1
            elif row["home_goals"] > row["away_goals"]:
                team2_wins += 1
            else:
                draws += 1
            team1_goals_scored += row["away_goals"]
            team1_goals_conceded += row["home_goals"]

    return {
        "h2h_wins": team1_wins,
        "h2h_losses": team2_wins,
        "h2h_draws": draws,
        "h2h_goals_scored": team1_goals_scored,
        "h2h_goals_conceded": team1_goals_conceded,
        "h2h_total_matches": len(recent_h2h_matches),
    }


def calculate_team_form(
    df: pd.DataFrame, team: str, end_date: datetime, recent_matches: int = 10
) -> Dict[str, int]:
    team_matches = df[
        (df["home_team"] == team) | (df["away_team"] == team)
    ].sort_values(by="match_date", ascending=False)
    recent_team_matches = team_matches[
        team_matches["match_date"] <= end_date
    ].head(recent_matches)

    if recent_team_matches.empty:
        return {
            "form_wins": 0,
            "form_losses": 0,
            "form_draws": 0,
            "form_goals_scored": 0,
            "form_goals_conceded": 0,
            "form_total_matches": 0,
        }

    wins = losses = draws = goals_scored = goals_conceded = 0

    for _, row in recent_team_matches.iterrows():
        if row["home_team"] == team:
            if row["home_goals"] > row["away_goals"]:
                wins += 1
            elif row["home_goals"] < row["away_goals"]:
                losses += 1
            else:
                draws += 1
            goals_scored += row["home_goals"]
            goals_conceded += row["away_goals"]
        else:  # team is away
            if row["away_goals"] > row["home_goals"]:
                wins += 1
            elif row["away_goals"] < row["home_goals"]:
                losses += 1
            else:
                draws += 1
            goals_scored += row["away_goals"]
            goals_conceded += row["home_goals"]

    return {
        "form_wins": wins,
        "form_losses": losses,
        "form_draws": draws,
        "form_goals_scored": goals_scored,
        "form_goals_conceded": goals_conceded,
        "form_total_matches": len(recent_team_matches),
    }


def calculate_home_form(
    df: pd.DataFrame, team: str, end_date: datetime, recent_matches: int = 5
) -> Dict[str, int]:
    home_matches = df[df["home_team"] == team].sort_values(
        by="match_date", ascending=False
    )
    recent_home_matches = home_matches[
        home_matches["match_date"] <= end_date
    ].head(recent_matches)

    if recent_home_matches.empty:
        return {
            "home_form_wins": 0,
            "home_form_losses": 0,
            "home_form_draws": 0,
            "home_form_goals_scored": 0,
            "home_form_goals_conceded": 0,
            "home_form_total_matches": 0,
        }

    wins = losses = draws = goals_scored = goals_conceded = 0

    for _, row in recent_home_matches.iterrows():
        if row["home_goals"] > row["away_goals"]:
            wins += 1
        elif row["home_goals"] < row["away_goals"]:
            losses += 1
        else:
            draws += 1
        goals_scored += row["home_goals"]
        goals_conceded += row["away_goals"]

    return {
        "home_form_wins": wins,
        "home_form_losses": losses,
        "home_form_draws": draws,
        "home_form_goals_scored": goals_scored,
        "home_form_goals_conceded": goals_conceded,
        "home_form_total_matches": len(recent_home_matches),
    }


def calculate_away_form(
    df: pd.DataFrame, team: str, end_date: datetime, recent_matches: int = 5
) -> Dict[str, int]:
    away_matches = df[df["away_team"] == team].sort_values(
        by="match_date", ascending=False
    )
    recent_away_matches = away_matches[
        away_matches["match_date"] <= end_date
    ].head(recent_matches)

    if recent_away_matches.empty:
        return {
            "away_form_wins": 0,
            "away_form_losses": 0,
            "away_form_draws": 0,
            "away_form_goals_scored": 0,
            "away_form_goals_conceded": 0,
            "away_form_total_matches": 0,
        }

    wins = losses = draws = goals_scored = goals_conceded = 0

    for _, row in recent_away_matches.iterrows():
        if row["away_goals"] > row["home_goals"]:
            wins += 1
        elif row["away_goals"] < row["home_goals"]:
            losses += 1
        else:
            draws += 1
        goals_scored += row["away_goals"]
        goals_conceded += row["home_goals"]

    return {
        "away_form_wins": wins,
        "away_form_losses": losses,
        "away_form_draws": draws,
        "away_form_goals_scored": goals_scored,
        "away_form_goals_conceded": goals_conceded,
        "away_form_total_matches": len(recent_away_matches),
    }


def build_features(
    historical_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    league_base_rates: Dict[str, float],
    include_results: bool = True,
) -> pd.DataFrame:
    """Construct the feature table used for training/prediction."""

    records: List[Dict[str, object]] = []
    source_df = historical_df if include_results else fixtures_df

    for _, row in source_df.iterrows():
        match_date = row["match_date"]
        league = row["league"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        h2h_stats = calculate_h2h_stats(historical_df, home_team, away_team, match_date)
        home_form = calculate_team_form(historical_df, home_team, match_date)
        away_form = calculate_team_form(historical_df, away_team, match_date)
        home_home_form = calculate_home_form(historical_df, home_team, match_date)
        away_away_form = calculate_away_form(historical_df, away_team, match_date)

        base_rate = league_base_rates.get(
            league,
            (historical_df["home_goals"] + historical_df["away_goals"] > 0).mean(),
        )

        record: Dict[str, object] = {
            "match_date": match_date,
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
            "league_base_rate": base_rate,
            **h2h_stats,
            **{f"home_{k}": v for k, v in home_form.items()},
            **{f"away_{k}": v for k, v in away_form.items()},
            **{f"home_home_{k}": v for k, v in home_home_form.items()},
            **{f"away_away_{k}": v for k, v in away_away_form.items()},
        }

        if include_results:
            record["home_goals"] = row["home_goals"]
            record["away_goals"] = row["away_goals"]
            record["total_goals_over_0_5"] = int(row["home_goals"] + row["away_goals"] > 0)

        records.append(record)

    return pd.DataFrame(records)


def proxy_for_missing_h2h(row: pd.Series) -> float:
    """Fallback probability estimate when head-to-head data is missing."""

    total_matches = row["home_form_total_matches"] + row["away_form_total_matches"]
    if total_matches > 0:
        total_goals = (
            row["home_form_goals_scored"]
            + row["home_form_goals_conceded"]
            + row["away_form_goals_scored"]
            + row["away_form_goals_conceded"]
        )
        return total_goals / total_matches
    return row["league_base_rate"]


def apply_h2h_proxy(features: pd.DataFrame) -> pd.DataFrame:
    """Replace empty H2H stats with a proxy derived from team form."""

    proxy_values = features.apply(proxy_for_missing_h2h, axis=1)
    for col in [
        "h2h_wins",
        "h2h_losses",
        "h2h_draws",
        "h2h_goals_scored",
        "h2h_goals_conceded",
    ]:
        if col in features.columns:
            features[col] = features.apply(
                lambda row: proxy_values[row.name]
                if row.get("h2h_total_matches", 0) == 0
                else row[col],
                axis=1,
            )
    return features.drop(columns=["h2h_total_matches"], errors="ignore")


def prepare_training_data(features_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric dtypes and handle any missing values for training."""

    numeric_df = features_df.copy()
    numeric_df.fillna(0, inplace=True)
    return numeric_df


def train_model(features_df: pd.DataFrame) -> CalibratedClassifierCV:
    """Train the gradient boosting model and calibrate it."""

    feature_columns = [
        col
        for col in features_df.columns
        if col
        not in {
            "match_date",
            "league",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "total_goals_over_0_5",
        }
    ]

    sorted_df = features_df.sort_values(by="match_date")
    X = sorted_df[feature_columns]
    y = sorted_df["total_goals_over_0_5"]

    split_index = int(len(sorted_df) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_val, y_val)

    return calibrated_model


def generate_predictions(
    calibrated_model: CalibratedClassifierCV,
    training_features: pd.DataFrame,
    fixture_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict probabilities for the upcoming fixtures."""

    feature_columns = [
        col
        for col in training_features.columns
        if col
        not in {
            "match_date",
            "league",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "total_goals_over_0_5",
        }
    ]

    X_fixture = fixture_features[feature_columns]
    fixture_features = fixture_features.copy()
    fixture_features["predicted_proba_over_0_5"] = calibrated_model.predict_proba(
        X_fixture
    )[:, 1]
    return fixture_features.sort_values(by="predicted_proba_over_0_5", ascending=False)


def format_justification(row: pd.Series) -> str:
    base_rate_pct = row["league_base_rate"] * 100
    home_avg = (
        row["home_form_goals_scored"] / row["home_form_total_matches"]
        if row["home_form_total_matches"]
        else 0
    )
    away_avg = (
        row["away_form_goals_scored"] / row["away_form_total_matches"]
        if row["away_form_total_matches"]
        else 0
    )
    return (
        f"League base over-0.5 rate {base_rate_pct:.1f}%, "
        f"recent home attack {home_avg:.2f} goals/match, "
        f"recent away attack {away_avg:.2f} goals/match."
    )


def print_predictions(predictions: pd.DataFrame) -> None:
    for _, row in predictions.iterrows():
        probability = row["predicted_proba_over_0_5"] * 100
        output_lines = [
            f"Match: {row['match_date'].strftime('%Y-%m-%d')} | {row['league']} | "
            f"{row['home_team']} vs {row['away_team']}",
            f"Predicted P(Over 0.5): {probability:.2f}%",
            f"Justification: {format_justification(row)}",
        ]

        if probability >= 98:
            output_lines.append("Threshold: Exceeds 98% threshold (Highly Likely)")
        elif probability >= 95:
            output_lines.append("Threshold: Exceeds 95% threshold (Likely)")

        print("\n".join(output_lines))
        print("-" * 50)


def main() -> None:
    random.seed(42)

    leagues = [
        LeagueConfig("English Premier League"),
        LeagueConfig("English Championship"),
        LeagueConfig("Italian Serie A"),
        LeagueConfig("German Bundesliga"),
        LeagueConfig("Spanish La Liga"),
        LeagueConfig("Portuguese Primeira Liga"),
    ]

    historical_df = create_historical_data(leagues)
    fixtures_df = create_current_fixtures(leagues)

    # Introduce inconsistencies and clean them.
    historical_df.loc[
        historical_df.sample(frac=0.01, random_state=42).index, "home_team"
    ] = "Manchester Utd"
    historical_df.loc[
        historical_df.sample(frac=0.005, random_state=99).index, "away_goals"
    ] = None

    team_name_mapping = {"Manchester Utd": "English Premier League Home Team 17"}
    historical_df["home_team"] = historical_df["home_team"].replace(team_name_mapping)
    historical_df["away_team"] = historical_df["away_team"].replace(team_name_mapping)

    historical_df["away_goals"].fillna(historical_df["away_goals"].median(), inplace=True)
    historical_df["home_goals"] = historical_df["home_goals"].astype(int)
    historical_df["away_goals"] = historical_df["away_goals"].astype(int)

    league_base_rates = (
        historical_df.groupby("league")[["home_goals", "away_goals"]]
        .apply(lambda x: (x["home_goals"] + x["away_goals"] > 0).mean())
        .to_dict()
    )

    features_df = build_features(historical_df, historical_df, league_base_rates)
    features_df = apply_h2h_proxy(features_df)
    features_df = prepare_training_data(features_df)

    fixture_features_df = build_features(
        historical_df, fixtures_df, league_base_rates, include_results=False
    )
    fixture_features_df = apply_h2h_proxy(fixture_features_df)
    fixture_features_df = prepare_training_data(fixture_features_df)

    calibrated_model = train_model(features_df)
    predictions = generate_predictions(
        calibrated_model, features_df, fixture_features_df
    )

    print_predictions(predictions)


if __name__ == "__main__":
    main()
