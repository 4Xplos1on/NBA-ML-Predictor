## Development Log: NBA ML Predictor

---

## Session 1 & 2: Data Sourcing & Network Limitations

**Objective:**  
Secure a reliable stream of daily NBA box score data to train the model.

---

### The Approach & Hurdles

**Web Scraping:**  
Initially built a scraper for Basketball-Reference using BeautifulSoup.  
Immediately hit a `403 Forbidden` wall due to Cloudflare anti-bot protections.

**Third-Party APIs:**  
Tried the balldontlie API, but the specific stats endpoint I needed was behind a paywall.

**Official NBA API:**  
Encountered `SSL: HANDSHAKE_FAILURE` due to school firewall restrictions.  
Bypassed with a VPN, but then hit silent timeout blocks on scripted traffic.

---

### The Pivot

Instead of spending weeks fighting network security, I pivoted to a static historical dataset to build and validate the core Machine Learning pipeline first.  
Live API integration will be reintroduced once the model logic is fully stable.

---

## Session 3: Feature Engineering & Baseline Model

**Objective:**  
Transform raw box scores into predictive signals and train a baseline XGBoost model.

---

### Engineering Hurdles

**Noise in Raw Data:**  
Raw stats like points and rebounds were too noisy.  
Engineered 5-game rolling differentials:

```
(Home Avg - Away Avg)
```

→ Allows the model to learn performance gaps instead of memorizing teams.

**Overfitting:**  
Initial model showed artificially high accuracy (memorization).

Applied constraints:
- `max_depth = 5`
- `min_samples_leaf = 20`

---

### Result

- `59.4%` cross-validation accuracy  
- Model serialized using `joblib` for fast inference

---

## Session 4: Building the Automated Processor (v2)

**Objective:**  
Build `processor.py` to transform raw historical logs into clean matchup differentials.

---

### Key Data Science Decision: EWMA vs Rolling Average

Used Exponentially Weighted Moving Average (EWMA):

```
lambda x: x.shift(1).ewm(span=5).mean()
```

**Why:**  
- Rolling averages treat all past games equally  
- EWMA prioritizes recent games → better momentum capture

**Critical Detail:**  
`.shift(1)` prevents data leakage by ensuring a game never includes its own stats.

---

### Engineering Hurdles

**Data Leakage Prevention:**  
Strict `.shift(1)` implementation to avoid fake accuracy.

**Pipeline Alignment:**  
- Synced team ID formats
- Handled NaN values at season starts

---

## Session 5: Live API Migration & Pipeline Hardening

**Objective:**  
Reconnect live NBA API, test real predictions, and build logging system.

---

### Engineering Hurdles

**API Deprecation:**  
`ScoreboardV2` became unreliable.  
→ Migrated to `ScoreboardV3` (nested JSON parsing required)

**Type-Casting Failures:**  
- API returns IDs as strings: `"1610612747"`
- CSV stores integers

**Fix:**  
```
int(team_id)
```

---

### Predictor Logic

- Engine named **NBA_Predictor**
- `THRESHOLD = 0.65`
  - `BET` → high-confidence picks
  - `PASS` → low-confidence games

**Anti-Duplicate Logger:**  
Prevents duplicate predictions using game ID checks.

---

## Session 6: Crossing the Vegas Baseline

**Objective:**  
Push model accuracy beyond ~65–67% Vegas baseline.

---

### New Features

**eFG% (Effective FG%)**  
Weights 3-pointers correctly.

**Fatigue Context (`IS_B2B`)**  
Back-to-back game indicator.

**Multi-Window Averages**  
- 5-game (short-term)
- 10-game (medium-term)

**Advanced Context**  
- `ELO_DIFF` → team strength
- `ALTITUDE_FLAG` → Denver/Utah impact

---

### Engineering Hurdle: Class Imbalance

Home teams win ~58% of games → model bias.

**Fix:** Dynamic class weighting

```python
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
```

→ Penalizes missed upsets more heavily

---

## Final Status

- Fully automated pipeline
- Stable live predictions
- No critical errors

**Final Holdout Accuracy: `69.0%`**  
→ Successfully exceeds Vegas baseline
