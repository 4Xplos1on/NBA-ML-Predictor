## Development Log: NBA ML Predictor

---

## Session 1: The Scraper Wall & Security Roadblocks

**Objective:**  
Build a training dataset by scraping daily NBA box scores from Basketball-Reference.

**What I Built:**  
`web-scraper(v.1)[Timeout].py` using `requests` and `BeautifulSoup`.

**The Issue:**  
Immediately hit a `403 Forbidden` error.  
The site uses Cloudflare to tarpit bots, making standard HTML scraping nearly impossible without advanced (and slow) bypasses.

**The Pivot:**  
Instead of fighting bot protections, I moved toward structured API data.  
It’s cleaner, more reliable, and allows the Surface Pro 9 to focus on logic rather than connection handling.

---

## Session 2: API Research & Network Bottlenecks

**Objective:**  
Use professional endpoints to retrieve structured JSON data.

**What I Built:**  
Two clients:
- `nba_api__(v.2)[Timeout].py` *(official `nba_api`)*  
- `balldontlie_api(v.2.1)[401].py` *(third-party API)*  

**The Issues:**

**Network Security:**  
Encountered `SSL: HANDSHAKE_FAILURE` due to school firewall interference.  
**Fix:** Used a VPN and `verify=False`.

**Official API:**  
Request accepted but no response returned → `ReadTimeoutError`.  
Likely silent blocking of scripted traffic.

**Third-Party API:**  
Connected successfully, but hit `401 Unauthorized`.  
Required `stats` endpoint is behind a paywall.

**The Pivot:**  
Switched to a static dataset (`Kaggle CSV`) to eliminate network constraints and focus fully on the ML pipeline.

---

## Session 3: Feature Engineering & Production

**Objective:**  
Move from raw data to a functional, automated prediction system.

---

### The Breakthroughs

**Differentials vs. Raw Stats:**  
Raw stats (`points`, `rebounds`) were too noisy.  
Switched to 5-game rolling differentials:  
`(Home Team Avg - Away Team Avg)`  

→ Model learns performance gap instead of memorizing team names.

**Fixing Overfitting:**  
Initial model showed inflated accuracy (memorization).  

Applied constraints:
- `max_depth = 5`  
- `min_samples_leaf = 20`  

→ Result: `59.4%` cross-validation accuracy.

---

### Production & Deployment

**Serialization:**  
Used `joblib` to freeze the trained model (`nba_model.pkl`).  
→ No retraining required  
→ Predictions run in milliseconds  

**Debugging:**  
- Fixed `ValueError` caused by feature mismatch  
- Synced training + inference features  

**File Pathing:**  
- Issue after moving to `/src`  
- Fixed using `os.path.join` + `BASE_DIR`

---

## Status

- System fully automated  
- Pipeline:  
  `Data → Features → Model → Predictions`  

- First successful prediction slate:  
  `April 6, 2026`
