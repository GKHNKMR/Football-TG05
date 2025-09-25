# Football-TG05

This repository contains a self-contained Python script that simulates football
fixtures, trains a gradient boosting model and predicts the probability of a
match finishing with more than 0.5 total goals.

## Requirements

The script depends on the scientific Python stack:

- Python 3.10+
- pandas
- scikit-learn

If the packages are not available in your environment you can install them via
`pip install pandas scikit-learn`.

## Usage

```bash
python over05_prediction.py
```

The program will:

1. Generate five and a half seasons of synthetic historical data for several
   major European leagues.
2. Engineer features such as head-to-head performance, recent form and league
   scoring rates.
3. Fit and calibrate a gradient boosting classifier on the simulated results.
4. Score a set of upcoming fixtures and display them ordered by the predicted
   probability of finishing with at least one goal.

Each fixture includes a short justification string and annotations when the
predicted probability surpasses 95% or 98%.
