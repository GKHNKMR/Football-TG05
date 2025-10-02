import pandas as pd
import numpy as np
from math import exp
from pathlib import Path

# Input / Output files
INPUT_FILE = "SQL_Fixtures_PL_CS_H2H_TR.xlsx"
OUTPUT_FILE = "predictions.xlsx"

# Sheets
PAST_SHEET = "PastMatches"   # <- geçmiş maçların olduğu sayfanın adı
FIXTURE_SHEET = "Fixture"    # <- 2025-26 fixture sayfasının adı

def calculate_team_stats(df):
    """Geçmiş maçlardan her takım için gol atma / yeme ortalamalarını çıkarır."""
    stats = {}

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg = row["HomeGoals"]
        ag = row["AwayGoals"]

        if home not in stats:
            stats[home] = {"scored": [], "conceded": []}
        if away not in stats:
            stats[away] = {"scored": [], "conceded": []}

        stats[home]["scored"].append(hg)
        stats[home]["conceded"].append(ag)
        stats[away]["scored"].append(ag)
        stats[away]["conceded"].append(hg)

    # Ortalama değerleri hesapla
    summary = {}
    for team, vals in stats.items():
        summary[team] = {
            "avg_scored": np.mean(vals["scored"]) if vals["scored"] else 1.2,
            "avg_conceded": np.mean(vals["conceded"]) if vals["conceded"] else 1.2,
        }
    return summary

def poisson_over05(lambda_home, lambda_away):
    """Poisson modeli ile Over 0.5 ihtimali"""
    lam = lambda_home + lambda_away
    return 1 - exp(-lam)

def main():
    # 1. Geçmiş maçları oku
    past_df = pd.read_excel(INPUT_FILE, sheet_name=PAST_SHEET)

    # 2. Takım istatistiklerini çıkar
    stats = calculate_team_stats(past_df)

    # 3. Fixture oku
    fixture_df = pd.read_excel(INPUT_FILE, sheet_name=FIXTURE_SHEET)

    gt05_values = []
    for _, row in fixture_df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Takım ortalamaları
        home_stats = stats.get(home, {"avg_scored": 1.2, "avg_conceded": 1.2})
        away_stats = stats.get(away, {"avg_scored": 1.2, "avg_conceded": 1.2})

        # Basit beklenen gol hesaplama
        lambda_home = (home_stats["avg_scored"] + away_stats["avg_conceded"]) / 2
        lambda_away = (away_stats["avg_scored"] + home_stats["avg_conceded"]) / 2

        p_over05 = poisson_over05(lambda_home, lambda_away)
        gt05_values.append(round(p_over05, 3))

    # 4. Fixture tablosuna GT05 sütunu ekle
    fixture_df["GT05"] = gt05_values

    # 5. Yeni Excel’e kaydet
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        past_df.to_excel(writer, sheet_name=PAST_SHEET, index=False)
        fixture_df.to_excel(writer, sheet_name=FIXTURE_SHEET, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

