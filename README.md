# NBA ML Predictor

**Goal:**  
A functional machine learning tool that predicts NBA game outcomes.


---

## Project Performance

- **Cross-Validation Accuracy:** `59.4%`  
- **Status:** Fully functional and automated for the 2026 season  
- **Verdict Logic:**  
  - Issues a `"BET"` verdict only when win probability > `65%`  
  - Focuses on high-confidence predictions over volume  

---

## Technical Overview

The system uses a `RandomForestClassifier` trained on historical NBA data from **2015 → 2026**.

To prevent overfitting:
- `max_depth = 5`  
- `min_samples_leaf = 20`  

→ Ensures the model generalizes instead of memorizing past games.

---

## Key Features

**Performance Differentials:**  
- Uses `(Home Team Avg - Away Team Avg)`  
- Captures relative strength instead of isolated stats  

**Rolling 5-Game Form:**  
- Focuses on most recent games  
- Captures momentum and short-term trends  

**Live Integration:**  
- Uses `nba_api` to fetch real-time matchups  
- Generates instant predictions  

---

## How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the Live Predictor**
```bash
python src/nba_live_predict.py
```

---

## Key Technical Insights

**Security & Tarpitting:**  
- Platforms use advanced protection
- Scraping attempts are often blocked or slowed intentionally  
- Switching to structured datasets significantly improved workflow  

**Network Constraints:**  
- Managed networks may intercept API traffic  
- Workarounds:
  - VPN routing  
  - `verify=False` for SSL handling  

**Differentials > Raw Stats:**  
- Comparing Team A vs Team B directly is more predictive  
- Relative performance ("gap") outperforms absolute metrics  

---

## Summary

- End-to-end ML pipeline operational  
- Automated daily predictions  
- Optimized for speed, stability, and realistic performance  
