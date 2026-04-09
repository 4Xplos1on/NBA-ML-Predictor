# NBA ML Predictor — Full Gemini Session Notes

Archive of the complete conversation with Gemini that walked through building the v1 model and planning v2. Kept for reference so the full reasoning, troubleshooting, and roadmap isn't lost.

---

## 1. SQLite Loading Issue (Kernel Hang, Not DB)

- SQLite `connect()` is essentially instant (<1 ms) — it's just opening a local file, no network.
- The ~60 second "hang" was the Jupyter kernel booting in VS Code after installing `pandas` / `ipykernel`, not the DB.
- Fix path: interrupt kernel → confirm `.venv (Python 3.14.2)` is selected top-right → Restart → run cell.
- When the entire notebook UI froze (couldn't even add cells), the Jupyter extension host had crashed. Fix: `Ctrl+Shift+P` → `Developer: Reload Window`, or fully close VS Code.

---

## 2. Raw Data Inspection

Dataset: `nba.sqlite`, table `game`. Raw row has columns like `season_id`, `team_id_home`, `team_abbreviation_home`, `game_id`, `game_date`, `matchup_home`, `wl_home`, `min`, `fgm_home`, ..., `pts_away`, `plus_minus_away`, `video_available_away`, `season_type`.

Seasons range: 1946 → 2023 (147 unique season IDs). Oldest game 1947-04-02, newest 2023-06-12.

### Critical issues identified
1. **Target format:** `wl_home` was string `'W'/'L'` — had to map to `1/0`.
2. **Massive data leakage:** raw box-score columns (`pts_home`, `fgm_home`, `pts_away`, etc.) are post-game stats. Leaving them in = model trivially learns "higher points = winner," useless for forecasting.
3. **Useless identifiers:** `game_id`, `video_available_away`, `matchup_home`, string names — drop.
4. **Ancient data:** basketball pre-shot-clock / pre-3pt-line is noise. Filter to modern era (`season_id >= '22015'`).

---

## 3. Feature Engineering Philosophy

### Why raw DB data isn't enough
The model needs a **pre-tip-off snapshot**. Every row must represent what was knowable 5 minutes before the game. Approach:

- **Rolling averages** — group by team, `.rolling(window=5).mean().shift(1)` on points/rebounds/assists/turnovers. The `.shift(1)` is critical: pushes the average down one row so a game's prediction uses only the *previous* 5 games, never itself.
- **Rest days (schedule fatigue)** — `groupby(team_id).game_date.diff().dt.days`. Back-to-backs (1 day) = tired team, upset risk.
- **Win streak / recent form** — captures momentum beyond just averages.
- **Head-to-head** — some teams match up poorly against specific opponents.

### Why `sklearn` toy datasets hide all this work
- Toy datasets (Iris, Titanic) are **pre-cleaned, i.i.d., pre-shuffled, leak-free**. Real DB data is chronological, contains answer-key columns, and needs row-level transformation.
- **No random train/test split** on time-series sports data — you'd train on 2024 to predict 2018, which is cheating. Must split sequentially by date/season.

### The "Differential" upgrade
Instead of feeding `home_pts_5g_avg` and `away_pts_5g_avg` as separate features, compute the **gap**:

```
pts_diff_5g = home_pts_5g_avg - away_pts_5g_avg
reb_diff_5g = home_reb_5g_avg - away_reb_5g_avg
rest_diff   = home_rest - away_rest
```

Collapses 8+ noisy features into 3 powerful signals. Reduces overfitting dramatically.

---

## 4. Model Iterations

### Iteration 1 — Logistic Regression baseline
- Features: 8 rolling averages (`pts/reb/ast/tov` × home/away) + 2 rest features.
- Dropped string/ID cols (`game_date`, `season_id`, `team_id_*`, `season_type`) before fitting.
- **Accuracy: 59.27%** — above the ~58% home-court baseline, so the model *is* learning something beyond "home team wins."
- Logistic Regression ≠ Linear Regression: LR predicts a probability (0–1) via a sigmoid curve, used for classification. Straight lines can't cap at 0/1.

### Iteration 2 — Random Forest (overfit)
- Same features, `n_estimators=100, min_samples_split=10`.
- **Accuracy: 56.26%** — *worse* than LR.
- Diagnosis via feature importance chart: importance spread almost equally across all 8 features → the forest is memorizing noise (bias-variance tradeoff, high variance side).

### Iteration 3 — Tuned Random Forest
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
```
- "Pruned" the trees: only 5 questions deep, pattern must exist in ≥20 games.
- **Accuracy: 57.40%** — recovery but still below LR. Ceiling reached with current features.

### Iteration 4 — Differentials + Tuned RF (final v1)
- Reduced to 3 features: `pts_diff_5g`, `reb_diff_5g`, `rest_diff`.
- Retrained on full `clean_df`.
- **Cross-validation accuracy: 59.44% (±small std)** via `cross_val_score(cv=5)`.

### Confusion matrix (v1 final)
- Strong home-court bias: 583 home wins correctly called, only 117 away wins correctly called, **444 missed away wins** (predicted home, away won).
- Interpretation: model is a "safe player" — leans on home advantage, struggles with upsets. Certainty score is the filter for this.

### High-confidence filtering
- `predict_proba()[:, 1]` gives the Home Win probability.
- Filter: `prob > 0.65 or prob < 0.35`.
- This is what lets the model be useful — overall 59% but much higher on the subset it's confident about.

---

## 5. Production Pipeline (v1)

### Serialization
```python
import joblib
joblib.dump(tuned_rf, 'nba_model.pkl')
```

### "Cheat sheet" export
Instead of re-querying SQLite at runtime, export the latest rolling averages per team into a flat CSV. Handled both home and away sides so each team has a single most-recent row:

```python
h_stats = nba_df[['team_id_home','home_pts_5g_avg','home_reb_5g_avg','game_date']].rename(...)
a_stats = nba_df[['team_id_away','away_pts_5g_avg','away_reb_5g_avg','game_date']].rename(...)
latest_stats = pd.concat([h_stats, a_stats]).sort_values('game_date').groupby('team_id').tail(1)
latest_stats.to_csv('latest_team_stats.csv', index=False)
```

### Live predictor script (`src/nba_live_predict.py`)

Key bits:

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'nba_model.pkl')
STATS_PATH = os.path.join(BASE_DIR, '..', 'data', 'latest_team_stats.csv')
```

Flow:
1. Load frozen model + stats CSV.
2. Hit `nba_api.live.nba.endpoints.scoreboard.ScoreBoard()` for today's games.
3. For each game, look up both teams in the cheat sheet (as `int` team_id).
4. Build a single-row DataFrame with the exact 3 feature names the model was trained on (critical — avoids "Feature names unseen at fit time" errors).
5. `predict_proba()` → certainty score.
6. Verdict: `BET` if `prob > 0.64 or prob < 0.36`, else `PASS`. Sort by certainty descending.

### First live run (April 6, 2026 slate)
```
Matchup                Certainty  Pick          Verdict
Magic @ Pelicans       67.4%      Pelicans      BET
Wizards @ Nets         65.5%      Nets          BET
Hornets @ Timberwolves 63.9%      Timberwolves  PASS
Raptors @ Celtics      63.8%      Celtics       PASS
Lakers @ Mavericks     62.9%      Mavericks     PASS
Jazz @ Thunder         56.9%      Thunder       PASS
Clippers @ Kings       55.9%      Kings         PASS
Pacers @ Cavaliers     54.3%      Cavaliers     PASS
Rockets @ Warriors     51.7%      Warriors      PASS
Grizzlies @ Bucks      46.9%      Grizzlies     PASS
Suns @ Bulls           38.9%      Suns          PASS
```
Note: these were the Sunday April 5 games — `nba_api` live endpoint returns the most recent slate until the next day's games tip off.

---

## 6. Debugging Journey (v1 production)

Errors hit and fixed, in order:

1. **`NameError: name 'X' is not defined`** on `cross_val_score(tuned_rf, X, y, cv=5)` — variable scope issue, defined `X_train/y_train` earlier but not `X/y`. Fix: explicitly define `X = clean_df[final_features]; y = clean_df['wl_home']`.
2. **`KeyError: "None of [['pts_diff_5g',...]] are in the [columns]"`** — diff columns never got created in `clean_df`. Fix: one consolidated "repair cell" that recalculates rolling averages, rest, AND differentials from scratch and drops NaNs.
3. **Feature mismatch on load** — old `.pkl` was trained on 10+ features, new script passing 3. Error: `"Feature names seen at fit time, yet now missing: ast_away_5g_avg, ..."`. Fix: retrain `tuned_rf` on only the 3 differential features, then `joblib.dump` again.
4. **`FileNotFoundError: nba_model.pkl`** when running from `src/` — hardcoded filename instead of using the `MODEL_PATH` variable. Fix: swap `joblib.load("nba_model.pkl")` → `joblib.load(MODEL_PATH)`. Same for `STATS_PATH`.
5. **`No such file or directory: 'src/nba_live_predict.py'`** — PowerShell was in `C:\Windows\system32`, not the project dir. Fix: `cd` into project first.
6. **Empty DataFrame output** — `stats_lookup` lookup was only checking `team_id_home`, so teams whose most-recent game was away got skipped. Fix: concat home+away rows, normalize to single `team_id` column, force `int` cast on both sides.
7. **UserWarning: X does not have valid feature names** — passing a plain list `[[diff]]` to `predict_proba`. Fix: build a `pd.DataFrame([{...}])` with exact column names matching training.
8. **Git: `fatal: not a git repository`** — workspace was downloaded as a ZIP, not cloned, so no `.git` folder. Fix: `git init` → `git remote add origin <URL>` → `git branch -M main`.
9. **`remote: Repository not found`** — trailing backslash in URL (`...Predictor.git\`). Fix: `git remote remove origin` → re-add clean URL → `git push -f origin main`.
10. **`pip: term not recognized`** on Windows PowerShell — Fix: `python -m pip install xgboost`.

---

## 7. Final v1 Repository Structure

```
NBA-ML-Predictor/
├── v1_legacy/              # Archived v1 work (Random Forest, static CSV)
│   ├── src/nba_live_predict.py
│   ├── data/latest_team_stats.csv
│   ├── models/nba_model.pkl
│   ├── notebooks/NBA_ML.ipynb
│   └── v1_deprecated/      # Failed scrapers (web-scraper, nba_api v2, balldontlie)
├── src/                    # v2.0 engine (to be built)
├── data/                   # v2.0 2-year dataset
├── models/                 # v2.0 XGBoost model
├── .gitignore              # *.sqlite to keep the huge DB local
├── requirements.txt        # pandas, scikit-learn, joblib, nba_api (+ xgboost for v2)
├── README.md
└── devlog.md
```

Key principles:
- **Relative paths only** (`BASE_DIR + '..' + folder + file`) so the project is portable.
- **`.gitignore` the SQLite** — too large for GitHub (100 MB limit).
- **Always run from the root** (`python src/nba_live_predict.py`) so the `..` in paths resolves correctly.

---

## 8. v2.0 Plan — XGBoost + Live NBA API

### Why v2
- v1 RF plateaued at ~59% because of limited feature space (only pts/reb diffs from 1-week static data).
- Real alpha comes from: bigger recent dataset, richer features (advanced ratings, rest, injuries), and a stronger algorithm.
- XGBoost = industry standard for tabular sports data. Builds trees sequentially, each correcting the previous tree's errors.

### v2 architecture
```
src/
├── downloader.py   # Bulk 2-3 year fetch via LeagueGameLog
├── processor.py    # Merge home/away rows, rolling averages, differentials
├── trainer.py      # XGBoost training + model save
└── predictor.py    # Live daily predictions
```

### `downloader.py` (already in progress)
```python
import pandas as pd, time
from nba_api.stats.endpoints import leaguegamelog

def download_history():
    seasons = ['2023-24', '2024-25', '2025-26']
    all_data = []
    for season in seasons:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season'
        )
        all_data.append(log.get_data_frames()[0])
        time.sleep(2)  # be a good citizen
    full = pd.concat(all_data, ignore_index=True)
    full.to_csv('data/nba_raw_2yr.csv', index=False)
```

Why `LeagueGameLog`: one request per season returns every team's every game. Much faster and less rate-limit-prone than pinging 30 `TeamGameLog` endpoints.

### Why 2-3 years is the sweet spot
- ~2,460 games over 2 full seasons, ~3,600+ with current season. Plenty for XGBoost.
- Modern NBA (pace, 3pt volume, roster turnover) shifts every few years — older data is polluted, not helpful.
- Lightweight on a Surface Pro 9.

### `processor.py` — the three filters (to be built)
1. **Single-row merge:** raw data has 2 rows per game. Split on `MATCHUP` (`vs.` = home, `@` = away), `pd.merge` on `Game_ID` with `_HOME`/`_AWAY` suffixes.
2. **Rolling momentum:** groupby team, rolling 5-game avg with `.shift(1)`, for PTS/REB/AST/PLUS_MINUS.
3. **Differentials:** `PTS_DIFF = PTS_HOME - PTS_AWAY`, same for each stat. Target: `TARGET = (WL_HOME == 'W').astype(int)`.

### `trainer.py` (XGBoost baseline config)
```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
model.save_model('models/nba_xgboost_v2.json')
```

### v2 target performance
59.4% → 62–64% range. In sports betting a ~3% jump is the difference between break-even and profitable.

### Future feature ideas (beyond v2 baseline)
- Advanced ratings: Offensive Rating, Defensive Rating, Net Rating (points per 100 possessions — accounts for pace).
- Injury flags (binary `is_star_out`) — scrape from Rotowire / NBA injury API.
- Vegas lines as a feature — lets the model learn where Vegas is mispriced rather than rebuilding their baseline from scratch.
- Travel / altitude (Denver specifically).
- Head-to-head recent record.

---

## 9. Tooling / Env Notes

- Python 3.14.2 in `.venv`.
- `pip install` on PowerShell: use `python -m pip install <pkg>`.
- Windows path gotcha: never use backslashes inside code strings for portability — `os.path.join` handles it.
- `nba_api` works with default headers as long as you `time.sleep(1-2)` between calls. No custom headers needed on current network. Custom headers / VPN / `verify=False` were only needed during the earlier scraping-heavy attempts.
- Model file format: `.pkl` (joblib) works for both RF and XGBoost. XGBoost also supports native `.json` save which is more portable across versions.

---

## 10. Status Snapshot

- **v1:** Complete. Random Forest + differentials, 59.4% CV accuracy, live predictor working against `nba_api` scoreboard, archived to `v1_legacy/`.
- **v2:** In progress. Repo refactored, XGBoost installed, raw data downloaded. Next: build `processor.py` → train XGBoost → replace predictor.

---

## 11. v2 Current State (Session 2 — April 2026)

### What's confirmed working
- **`nba_api` works with no custom headers** — tested a clean `PlayerGameLog` request for LeBron James (player_id `2544`, season `2023-24`) using only library defaults + `timeout=30`. Succeeded instantly. No VPN, no `verify=False`, no spoofed User-Agent needed on current network.
- **`LeagueGameLog` bulk pull complete** — `data/nba_raw_2yr.csv` downloaded and confirmed: **7,294 rows** covering three seasons (`2023-24`, `2024-25`, `2025-26`), Regular Season only.

### Raw CSV columns (from `nba_api` LeagueGameLog)
```
SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID, GAME_DATE,
MATCHUP, WL, MIN, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA,
FT_PCT, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, PLUS_MINUS,
VIDEO_AVAILABLE
```
Sample row: `22023,1610612743,DEN,Denver Nuggets,0022300061,2023-10-24,DEN vs. LAL,W,...,119,12,1`

### Key structural detail
Each game produces **two rows** — one for each team (home team has `vs.` in MATCHUP, away has `@`). The `processor.py` must merge these into one row per game using `GAME_ID` as the join key.

### Current file inventory in `src/`
| File | Status | Purpose |
|------|--------|---------|
| `nba_api-datareq.py` | ✅ Complete | Downloads 3-season bulk data (saves as `nba_game_logs.csv`; the `nba_raw_2yr.csv` in `data/` is the actual output used) |
| `processor.py` | 🔲 Empty | Needs to be built: merge, rolling avgs, differentials |

### `nba_api-datareq.py` current code
```python
from nba_api.stats.endpoints import leaguegamelog
import pandas as pd, time

def download():
    seasons = ['2023-24', '2024-25', '2025-26']
    all_seasons_data = []
    for season in seasons:
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star='Regular Season'
            )
            all_seasons_data.append(log.get_data_frames()[0])
            time.sleep(2)
        except Exception as e:
            print(f"Error downloading data for season {season}: {e}")
    return all_seasons_data

if __name__ == "__main__":
    data_list = download()
    if data_list:
        full_data = pd.concat(data_list, ignore_index=True)
        full_data.to_csv('nba_game_logs.csv', index=False)
        print("Total rows downloaded:", len(full_data))
```

---

## 12. `processor.py` — What Needs to Be Built

The raw CSV has 2 rows per game. `processor.py` must:

### Step 1 — Split by home/away
```python
home = df[df['MATCHUP'].str.contains('vs.')].copy()
away = df[df['MATCHUP'].str.contains('@')].copy()
```

### Step 2 — Merge into one row per game
```python
merged = pd.merge(
    home, away,
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)
```

### Step 3 — Sort chronologically and compute rolling averages
Group by `TEAM_ID`, sort by `GAME_DATE`, then for each stat:
```python
merged['PTS_HOME_5G'] = merged.groupby('TEAM_ID_HOME')['PTS_HOME'].transform(
    lambda x: x.rolling(5).mean().shift(1)
)
```
Apply same to `REB`, `AST`, `PLUS_MINUS` for both sides.

### Step 4 — Compute differentials
```python
merged['PTS_DIFF']       = merged['PTS_HOME_5G'] - merged['PTS_AWAY_5G']
merged['REB_DIFF']       = merged['REB_HOME_5G'] - merged['REB_AWAY_5G']
merged['PLUS_MINUS_DIFF'] = merged['PLUS_MINUS_HOME_5G'] - merged['PLUS_MINUS_AWAY_5G']
```

### Step 5 — Target variable + clean up
```python
merged['TARGET'] = (merged['WL_HOME'] == 'W').astype(int)
clean = merged.dropna(subset=['PTS_DIFF', 'REB_DIFF', 'TARGET'])
```

### Key feature list for XGBoost (starting point)
- `PTS_DIFF` — 5-game rolling points gap (home minus away)
- `REB_DIFF` — rebounding gap
- `PLUS_MINUS_DIFF` — overall efficiency gap (hidden gem: encodes offense + defense together)
- `REST_DIFF` — days since last game (home minus away), optional add

---

## 13. XGBoost Trainer (`trainer.py`) — Plan

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, pandas as pd

df = pd.read_csv('data/processed_matchups.csv')  # output of processor.py

features = ['PTS_DIFF', 'REB_DIFF', 'PLUS_MINUS_DIFF']  # expand as needed
X = df[features]
y = df['TARGET']

# Chronological split — do NOT shuffle sports data
split_idx = int(len(df) * 0.85)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {acc*100:.2f}%")

# Save
model.save_model('models/nba_xgboost_v2.json')
# Or: joblib.dump(model, 'models/nba_xgboost_v2.pkl')
```

Target: break 62%. PLUS_MINUS_DIFF is expected to be the most predictive single feature (it encodes both offensive and defensive quality in one number).

---

## 14. Pending Items for Next Session

| Item | Priority | Notes |
|------|----------|-------|
| Write `processor.py` | 🔴 High | Core blocker — nothing else works without it |
| Run `trainer.py` | 🔴 High | Get v2 baseline accuracy number |
| Update README + devlog | 🟡 Medium | Remove old "security struggle" framing; the API works fine now |
| Draft LinkedIn post | 🟡 Medium | Richard said "next time we write that LinkedIn post" — to do next session |
| Update `requirements.txt` | 🟢 Low | Add `xgboost` to the file |

### LinkedIn post outline (to draft next session)
- Hook: "Can machine learning beat the bookies?" / building an NBA predictor
- Technical mention: v1 RF → v2 XGBoost upgrade, moving from static CSV to live API
- Key insight: differentials > raw stats; momentum > season averages
- Stat: 59.4% CV accuracy (honest, above home-court baseline)
- Visual: Feature Importance chart
- Link: GitHub repo

---

## 15. README / devlog — Planned Updates

The current README still has the "Security & Tarpitting" and "Network Constraints" sections — these should be removed since the API works normally. Planned replacement:

**Remove from README:**
- "Security & Tarpitting" bullet
- "Network Constraints" bullet (VPN, `verify=False`)

**Add to README:**
```
**Official API Integration:**
- Interfaces directly with `stats.nba.com` via the `nba_api` library.
- Implements smart request throttling (`time.sleep`) to ensure stable connectivity.
- Pulls real-time 5-game rolling snapshots — no static CSV dependency.
```

**Add to devlog Session 3:**
```
April 2026 — v2 Init:
- Confirmed nba_api works with default library headers, no workarounds needed.
- Downloaded 7,294 rows of Regular Season data (2023–2026) via LeagueGameLog.
- Repo restructured: v1 archived to v1_legacy/, fresh src/data/models/ ready for XGBoost.
- XGBoost installed via `python -m pip install xgboost`.
- Next: processor.py → trainer.py → v2 live predictor.
```
