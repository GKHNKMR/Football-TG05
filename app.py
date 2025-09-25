"""Flask web interface for the over 0.5 goal probability simulator."""
from __future__ import annotations

from flask import Flask, render_template, request

from over05_prediction import generate_prediction_table

app = Flask(__name__)


def _build_rows(predictions_df):
    rows = []
    for _, row in predictions_df.iterrows():
        threshold_label = row.get("threshold_label")
        if not isinstance(threshold_label, str) or not threshold_label.strip():
            threshold_label = None
        rows.append(
            {
                "match_date": row["match_date"].strftime("%Y-%m-%d"),
                "league": row["league"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "probability": f"{row['probability_pct']:.2f}",
                "justification": row["justification"],
                "threshold_label": threshold_label,
            }
        )
    return rows


@app.route("/", methods=["GET", "POST"])
def index():
    seed = request.form.get("seed", type=int)
    if seed is None:
        seed = 42

    predictions_df = generate_prediction_table(seed)
    rows = _build_rows(predictions_df)

    return render_template("index.html", predictions=rows, seed=seed)


if __name__ == "__main__":
    app.run(debug=True)
