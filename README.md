# NBA ML Predictor

**Goal:** Build a Machine Learning model to predict NBA game outcomes. This log tracks my development process, challenges, and architectural decisions.

---

### Session 1: Initial Scraper & Security Roadblocks
* **Objective:**    Scrape daily NBA box scores from Basketball-Reference to build a training dataset.
* **What I Built:** A Python web scraper, `WebScraper-v.1`, using `requests` and `BeautifulSoup` to target HTML game containers.
* **The Issue:**    The script returned an empty list. Checking the server response revealed a `403 Forbidden` error because the website uses Cloudflare to block automated bots.
* **The Fix:**      Instead of trying to bypass security, I will focus on migrating to the official Python `nba_api` library to securely and reliably pull structured data.

---

### Session 2: 
* **Objective:**
* **What I Built:**
* **The Issue:**
* **The Fix:**
       
