# LaLiga Season Finish Predictor

An XGBoost-based machine learning model that predicts the probability of a LaLiga team finishing in the **top 4** or winning the **title (top 1)** based on in-season performance statistics. Trained on 20 seasons of historical LaLiga data from [football-data.co.uk](https://www.football-data.co.uk/).

The best performing variant is **top4 with a 20-game rolling window**.

---

## How it works

The pipeline has three stages:

1. **Data prep** (`refdata.py`) — reads raw match CSVs, engineers features, and outputs a JSON training file
2. **Data** (`data.py`) — reads the most recent season's raw match CSV and outputs a JSON file with the same features as the training data
3. **Training and predicting** (`model/model1.py`) — runs grid search cross-validation, saves the best model, and at the end outputs predictions using the most recent league data from `data.py`. It is also possible to make predictions in a separate file since the model and its parameters are saved.

---

## Data setup

Download the LaLiga season files (`SP1.csv`) from [football-data.co.uk](https://www.football-data.co.uk/spainm.php) and place them in a local folder. Name them sequentially:

```
SP1(1).csv    ← the latest finished season
SP1(2).csv
...
SP1(20).csv   ← oldest season
```

Update the path at the top of `refdata.py` to point to your folder, then run:

```bash
python refdata.py
```

This outputs the training JSON.

For the current season, update the path at the top of `data.py` and run it. The script will output the data which the model will use to predict. Place the current season JSON in `laliga2526/`.

**Important note:** `refdata.py` and `data.py` are set up to predict who will win the league, with an emphasis on recent performance. To change this:

- From top 1 to top 4: in `refdata.py` go to line 60 and change `table['top1'] = (table['rank'] <= 1).astype(int)` to `table['top4'] = (table['rank'] <= 4).astype(int)`. In `model1.py` change `y = df["top1"]` to `y = df["top4"]`.
- From short term to long term: in `refdata.py` go to lines 41 and 53 and change the number in `.tail()` to however many games have already been played in the current season, and update the column name accordingly. In `model1.py` update the feature names `"Win_rate10"` and `"Wstrk_10"` to match.

---

## Training

Run the model training script:

```bash
python model/model1.py
```

This runs a grid search over hyperparameters, prints the test AUC score, saves the best model to `model/`, and outputs predictions so you can see the results immediately.

---

## Prediction

Run predictions for the current season:

```bash
python data.py
```

---

## Project structure

```
Laliga-prediction-model/
├── refdata.py              # raw CSV → feature JSON pipeline
├── data.py                 # current season data prep and prediction
├── model/
│   └── model1.py           # training + grid search
├── traingingdata/          # engineered training JSON
├── laliga2526/             # current season data
└── README.md
```

---

## Feature explanations

Features were chosen to capture three things: **attacking quality**, **defensive exposure**, and **recent form** — without leaking the final league position into the model.

### Shooting efficiency

| Feature | Description | Why it matters |
|--------|-------------|----------------|
| `SoT%` | Shots on target / total shots | Measures shot quality — teams that shoot accurately are more dangerous regardless of volume |
| `Gsh` | Goals per shot | Finishing efficiency — separates clinical teams from wasteful ones |
| `GSoT` | Goals per shot on target | Conversion rate once the shot is on target — strong proxy for striker quality |

### Volume per game

Per-game stats are used instead of raw totals to remove the effect of games played, making mid-season snapshots comparable to end-of-season ones.

| Feature | Description |
|--------|-------------|
| `G90` | Goals scored per game |
| `S90` | Shots attempted per game |
| `SoT90` | Shots on target per game |
| `C90` | Corners won per game |
| `AG90` | Goals conceded per game |
| `AS90` | Shots conceded per game |
| `ASoT90` | Shots on target conceded per game |
| `F90` | Fouls committed per game |
| `Y90` | Yellow cards per game |
| `R90` | Red cards per game |

Defensive stats (`AG90`, `AS90`, `ASoT90`) matter as much as attacking ones — top 4 teams in LaLiga consistently have among the lowest goals conceded in the league.

### Expected goals proxies

| Feature | Formula | Why it matters |
|--------|---------|----------------|
| `proxy_xG` | `TST × 0.3 + (TS - TST) × 0.05` | Approximates expected goals for — weighted by shot quality (on target vs off) |
| `proxy_xGA` | `ASoT × 0.3 + (AS - ASoT) × 0.05` | Approximates expected goals against — captures defensive vulnerability |

These are proxy values since full xG data is not available in the football-data.co.uk dataset. The weights (0.3 for on-target, 0.05 for off-target) reflect typical conversion rate differences between shot types. Since accessing these advanced statistics directly is not feasible, these approximations introduce some degree of uncertainty into the model.

### Recent form

| Feature | Description | Why it matters |
|--------|-------------|----------------|
| `Win_rate10` / `Win_rate20` | Win rate over last 10 or 20 games | Captures current momentum rather than just season average |
| `Wstrk_10` / `Wstrk_20` | Longest winning streak in last 10 or 20 games | Identifies teams on a run — a strong predictor of sustained top 4 pushes |

Two window sizes are trained separately (10 and 20 games) because shorter windows reflect hot form while longer windows reflect consistency. The 20-game window is the best performing variant overall.

### Season

The `season` feature (encoded as a number) allows the model to account for gradual changes in the league over time — for example shifts in overall competitiveness or the dominance of certain clubs across eras.

---

## What was deliberately excluded

- **Points total and league position** — these directly encode the outcome and would cause data leakage
- **Win / Draw / Loss raw counts** — replaced by per-game rates and form windows which are more informative
- **Betting odds columns** — stripped from the raw data entirely as they are not causal features
- **Team name** — not used as a feature since the model should generalise to promoted teams and not simply memorise that Real Madrid usually finishes top

---

## Requirements

```
xgboost
pandas
scikit-learn
matplotlib
numpy
```
