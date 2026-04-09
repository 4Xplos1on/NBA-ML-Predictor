# NBA ML Predictor — Dakota

---

**Goal:**  
Build a machine learning pipeline that predicts NBA game outcomes — and actually test whether it works against real Vegas lines.

## TL;DR / Executive Summary

**What it is:**  
An automated, end-to-end Machine Learning pipeline that predicts NBA game outcomes.

**The Engine:**  
An XGBoost classification model trained on:
- Rolling differentials  
- Elo ratings  
- Fatigue context (back-to-backs, rest days)

**The Result:**  
`69.0%` holdout accuracy — exceeding the Vegas baseline (~65–67%) by identifying inefficiencies in scheduling fatigue and shooting efficiency.

**Tech Stack:**  
- Python  
- Pandas  
- XGBoost  
- Scikit-Learn  
- NBA API

---

## Project Status

| Version | Algorithm | Accuracy | Status |
|---------|-----------|----------|--------|
| v1 | Random Forest | 59.4% CV | Complete → archived to `v1_legacy/` |
| v2 (Dakota) | XGBoost | 69% holdout | Active |

**Verdict Logic:**  
- Issues a `"BET"` verdict only when win probability > `65%`  
- Focuses on high-confidence predictions over volume  
- Logs all picks to `predictions_log.csv` with duplicate prevention  

---

## How It Started — v1

v1 was a basic Random Forest trained on a static Kaggle CSV. It had hard-coded absolute paths, no live prediction, and used `ScoreboardV2` for game data — which is now deprecated for 2025-26 season data. It got to 59.4% cross-validation accuracy before being archived.

v2 (Dakota) is a full rebuild: live data from the NBA API, XGBoost, relative pathing, a proper prediction engine, and a logging system to track real-world accuracy over time.

---

## Technical Overview

**v2 Pipeline:**  
`nba_api-datareq.py` → `processor.py` → `nba-predict_v2.py` → `predictions_log.csv`

**Data:** 3 seasons of live game logs pulled from `stats.nba.com` via `LeagueGameLog` (2023-24, 2024-25, 2025-26). Raw data: ~7,300 rows. Processed matchups: ~3,500 rows.

**Model:** `XGBClassifier` — 300 trees, learning rate 0.03, max depth 5. `scale_pos_weight` dynamically calculated from class balance to prevent the model from defaulting to "home win" on every prediction.

**Live Prediction:** Uses `ScoreboardV3` (migrated from deprecated `ScoreboardV2`) to fetch today's games, looks up each team's rolling stats from the processed CSV, and outputs a probability + BET/PASS verdict.

---

## Features

| Feature | What It Captures |
|---------|-----------------|
| `PTS_DIFF` | Scoring gap, 5-game EWMA |
| `REB_DIFF` | Rebounding gap, 5-game EWMA |
| `AST_DIFF` | Ball movement gap, 5-game EWMA |
| `TOV_DIFF` | Turnover discipline gap, 5-game EWMA |
| `FG_PCT_DIFF` | Shooting efficiency gap, 5-game EWMA |
| `FG3_PCT_DIFF` | 3-point shooting gap, 5-game EWMA |
| `PLUS_MINUS_DIFF` | Net scoring gap (offense + defense in one number) |
| `STL_DIFF` | Defensive pressure gap |
| `BLK_DIFF` | Paint protection gap |
| `*_10G_DIFF` | Same stats over 10-game window (medium-term form) |
| `REST_DAYS_DIFF` | Rest advantage — back-to-backs heavily impact performance |

Total: 19 features across two rolling windows + rest.

---

## Key Technical Insights

**Differentials > Raw Stats:**  
Comparing Team A vs Team B directly is more predictive than isolated numbers. Relative performance captures the actual matchup.

**EWMA > Simple Rolling:**  
Exponentially Weighted Moving Average weights recent games more heavily. A team that just went 4-1 after a cold streak reads differently than a team that went 4-1 two months ago.

**Chronological Splits:**  
Sports data cannot be randomly shuffled for train/test splits. Training on future games to predict the past is data leakage — split by date instead.

**`.shift(1)` Prevents Leakage:**  
The rolling average is shifted forward one row so a game never includes its own stats when the model trains. Without this, accuracy looks inflated but the model is cheating.

**ScoreboardV3 vs V2:**  
V3 returns nested JSON (`['scoreboard']['games']`) instead of a flat dataframe. Required rewriting the team ID lookup loop entirely.

**String vs Integer IDs:**  
The NBA API returns team IDs as strings. The processed CSV stores them as integers. Without explicit `int()` casting, lookups silently fail with "Matchup stats not found."

---

## How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Download / refresh raw data**
```bash
python src/nba_api-datareq.py
```

**3. Process into matchups**
```bash
python src/processor.py
```

**4. Run Dakota (live predictor)**
```bash
python src/nba-predict_v2.py
```

Or use the hub:
```bash
python src/main-app.py
```

---

## Real-World Context

Dakota started as a contrarian model — flagging picks on the Wizards and Celtics in April 2026 that Vegas was fading due to resting players. After expanding the feature set to include eFG%, win streaks, back-to-back flags, ELO ratings, and a 10-game rolling window, the model crossed the Vegas baseline.

Vegas consensus accuracy: ~65–67%  
Dakota holdout accuracy: **69%**

---

## Summary

- End-to-end ML pipeline operational  
- Live predictions with BET/PASS verdict and confidence threshold  
- Persistent logging with duplicate prevention  
- v1 archived, v2 (Dakota) active  
- Benchmark: 69% holdout — above the Vegas 65–67% baseline
