# NBA ML Predictor

An end-to-end machine learning pipeline that predicts NBA game outcomes using rolling team differentials, Elo ratings, and fatigue context. Trained on three seasons of live data from the NBA API, it issues high-confidence betting verdicts against a 65% probability threshold.

Holdout accuracy: **69.0%** (Vegas consensus sits around 65-67% for straight-up winner prediction).

**Built with:** Python, Pandas, XGBoost, scikit-learn, NBA API

---

## How It Works

**Pipeline:** `nba_api-datareq.py` feeds into `processor.py`, which feeds into `nba-predict_v2.py`, with results saved to `predictions_log.csv`.

1. **Data pull** -- Three seasons of game logs (2023-24 through 2025-26) fetched from stats.nba.com via `LeagueGameLog`. About 7,300 raw rows become about 3,500 processed matchups.

2. **Feature engineering** -- For each matchup, the pipeline computes rolling differentials (home minus away) across 19 features using two time windows:

   | Feature | What it captures |
   |---------|-----------------|
   | `PTS_DIFF` | Scoring gap, 5-game EWMA |
   | `REB_DIFF` | Rebounding gap, 5-game EWMA |
   | `AST_DIFF` | Ball movement gap, 5-game EWMA |
   | `TOV_DIFF` | Turnover discipline, 5-game EWMA |
   | `FG_PCT_DIFF` | Shooting efficiency, 5-game EWMA |
   | `FG3_PCT_DIFF` | 3-point shooting, 5-game EWMA |
   | `PLUS_MINUS_DIFF` | Net scoring, offense and defense combined |
   | `STL_DIFF` | Defensive pressure |
   | `BLK_DIFF` | Paint protection |
   | All 10-game variants | Same stats over a 10-game window |
   | `REST_DAYS_DIFF` | Rest advantage, back-to-backs matter |

3. **Model** -- `XGBClassifier` with 300 trees, learning rate 0.03, max depth 5. `scale_pos_weight` is calculated dynamically from class balance to counteract the 58% home-win base rate.

4. **Prediction** -- Fetches today's games via `ScoreboardV3`, looks up each team's current rolling stats, and outputs a win probability. Games above the 65% confidence threshold get a BET verdict; everything else gets PASS. All picks are logged to `predictions_log.csv` with duplicate prevention.

---

## Technical Decisions Worth Noting

**Differentials over raw stats.** Comparing Team A vs Team B directly is more predictive than feeding isolated team numbers. The model learns matchup dynamics, not team identities.

**EWMA over simple rolling averages.** Exponentially weighted moving averages put more weight on recent games. A team coming off a 4-1 streak reads differently than one that went 4-1 two months ago.

**Chronological train/test split.** Sports data is time-ordered. Random shuffling for train/test would let the model train on future games to predict the past. That's data leakage, not accuracy.

**`.shift(1)` to prevent same-game leakage.** Every rolling average is shifted forward one row so a game's own stats never appear in its own features. Without this, accuracy looks great but the model is cheating.

**Dynamic class weighting.** Home teams win 58% of NBA games. Without correction, the model learns to just predict "home win" every time. The positive class weight is set to the ratio of negative to positive samples, which penalizes missed upsets proportionally.

**ScoreboardV3 migration.** The original `ScoreboardV2` endpoint was deprecated for 2025-26 data. V3 returns nested JSON instead of flat DataFrames, which required rewriting the game-fetching logic. Team IDs also come back as strings instead of integers, and without explicit casting, lookups silently fail.

---

## Setup and Usage

Live predictions require an active NBA season. Historical model training works year-round, but the prediction engine depends on `ScoreboardV3` returning today's schedule. During the offseason or if the NBA API is rate-limiting, live predictions may return empty results or errors.

Currently Windows only. Cross-platform support is planned.

### Requirements

- Python 3.10 or higher
- Check "Add Python to PATH" during installation

### Steps

```
cd "C:\path\to\NBA-ML-Predictor-main"

python main.py
```

Select option 1 (SETUP) first. This installs dependencies and trains the model. After that, use option 2 to re-sync data and retrain, option 3 to check yesterday's picks, option 4 to predict tonight's games, or option 5 to exit.

The trained model is not included in the repo. Always run SETUP first.

---

## Project History

This started as a Random Forest on a static Kaggle CSV with 59.4% cross-validation accuracy. That version is archived in `v1_legacy/`. The current version is a full rebuild with live API data, XGBoost, proper feature engineering, and a prediction logging system.
