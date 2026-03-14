# Laliga-prediction-model
# LaLiga Season Finish Predictor

An XGBoost-based machine learning model that predicts the probability of a LaLiga team finishing in the **top 4** or winning the **title (top 1)** based on in-season performance statistics. Trained on 20 seasons of historical LaLiga data from [football-data.co.uk](https://www.football-data.co.uk/).

The best performing variant is **top4 with a 20-game rolling window**.

---

## How it works

The pipeline has three stages:

1. **Data prep** (`data_prep/refdata.py`) — reads raw match CSVs, engineers features, and outputs a JSON training file
2. **Training** (`train.py`) — runs grid search cross-validation across 4 model variants and saves the best model for each
3. **Prediction** (`predict.py`) — loads a saved model and outputs top 4 / title probabilities for the current season

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/laliga-predictor.git
cd laliga-predictor
pip install -r requirements.txt
```

---

## Data setup

Download the LaLiga season files (`SP1.csv`) from [football-data.co.uk](https://www.football-data.co.uk/spainm.php) and place them in `data/raw_data/`. Name them sequentially:

```
data/raw_data/SP1(1).csv    ← most recent season
data/raw_data/SP1(2).csv
...
data/raw_data/SP1(20).csv   ← oldest season
```

Then run the data prep script to generate the training file:

```bash
python data_prep/refdata.py
```

This outputs `data/training_data/20Y_LaLiga_data.json`.

For the current season, build a separate JSON in the same format and place it at:

```
data/current_season/2526_LaLiga.json
```

---

## Training

To train all 4 model variants (top1/top4 × window10/window20):

```bash
python train.py
```

This runs a grid search for each variant, prints the test AUC score, and saves the best model to `model/saved/`. A summary of all scores is saved to `model/saved/results.json`.

To train a specific variant only, edit the `MODELS` list in `config.py`.

---

## Prediction

To run predictions for the current season across all 4 variants:

```bash
python predict.py
```

Example output for the top4 model:

```
--- top4_w20 predictions ---
        Team  prob%
 Real Madrid   81.7
   Barcelona   55.0
  Ath Madrid   12.1
  Villarreal    9.5
      Girona    3.2
         ...
```

Probabilities are normalised to sum to 100% across the league. Teams below 1% are shown as 0%.

---

## Project structure

```
LaLiga_Pred/
├── config.py                  # model variants and feature lists
├── train.py                   # training + grid search
├── predict.py                 # load model and predict current season
├── data_prep/
│   └── refdata.py             # raw CSV → feature JSON pipeline
├── model/
│   └── saved/
│       ├── top1_w10.json
│       ├── top1_w20.json
│       ├── top4_w10.json
│       ├── top4_w20.json
│       └── results.json       # AUC scores for each variant
├── data/
│   ├── raw_data/              # raw CSVs — not committed
│   ├── training_data/         # engineered training JSON — not committed
│   └── current_season/        # current season JSON — not committed
├── requirements.txt
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

### Volume per 90 minutes

Per-90 stats are used instead of raw totals to remove the effect of games played, making mid-season snapshots comparable to end-of-season ones.

| Feature | Description |
|--------|-------------|
| `G90` | Goals scored per 90 minutes |
| `S90` | Shots attempted per 90 minutes |
| `SoT90` | Shots on target per 90 minutes |
| `C90` | Corners won per 90 minutes |
| `AG90` | Goals conceded per 90 minutes |
| `AS90` | Shots conceded per 90 minutes |
| `ASoT90` | Shots on target conceded per 90 minutes |
| `F90` | Fouls committed per 90 minutes |
| `Y90` | Yellow cards per 90 minutes |
| `R90` | Red cards per 90 minutes |

Defensive stats (`AG90`, `AS90`, `ASoT90`) matter as much as attacking ones — top 4 teams in LaLiga consistently have among the lowest goals conceded in the league.

### Expected goals proxies

| Feature | Formula | Why it matters |
|--------|---------|----------------|
| `proxy_xG` | `TST × 0.3 + (TS - TST) × 0.05` | Approximates expected goals for — weighted by shot quality (on target vs off) |
| `proxy_xGA` | `ASoT × 0.3 + (AS - ASoT) × 0.05` | Approximates expected goals against — captures defensive vulnerability |

These are proxy values since full xG data is not available in the football-data.co.uk dataset. The weights (0.3 for on-target, 0.05 for off-target) reflect typical conversion rate differences between shot types.

### Recent form

| Feature | Description | Why it matters |
|--------|-------------|----------------|
| `Win_rate10` / `Win_rate20` | Win rate over last 10 or 20 games | Captures current momentum rather than just season average |
| `Wstrk_10` / `Wstrk_20` | Longest winning streak in last 10 or 20 games | Identifies teams on a run — a strong predictor of sustained top 4 pushes |

Two window sizes are trained separately (10 and 20 games) because shorter windows reflect hot form while longer windows reflect consistency. The 20-game window is the best performing variant overall.

### Season

The `season` feature (encoded as a number) allows the model to account for gradual changes in the league over time — for example shifts in the overall competitiveness or the dominance of certain clubs across eras.

---

## What was deliberately excluded

- **Points total and league position** — these directly encode the outcome and would cause data leakage
- **Win / Draw / Loss raw counts** — replaced by per-90 rates and form windows which are more informative
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
