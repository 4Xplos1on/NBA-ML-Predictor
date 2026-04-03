## Session 1: Initial Scraper & Security Roadblocks

Objective: Scrape daily NBA box scores from Basketball-Reference to build a training dataset.

What I Built: `web-scraper(v.1)[Timeout].py`  
A Python web scraper using `requests` and `BeautifulSoup` to target HTML containers.

The Issue:  
The script returned a `403 Forbidden` error. The website uses Cloudflare to block automated bots from reading the HTML "skin" of the site.

Next steps / fix:  
Instead of trying to bypass security, I decided to migrate to structured API data to avoid messy HTML parsing and connection blocks.

---

## Session 2: API Research & Networking Bottlenecks

Objective: Use professional API endpoints to retrieve structured JSON data for player stats.

What I Built:  
Developed two API clients:  
- `nba_api__(v.2)[Shadowban].py` (using the official `nba_api` library)  
- `balldontlie_api(v.2.1)[Paywall required].py` (using a third-party developer API)

The Issue:

Network Security:  
Encountered an `SSL: HANDSHAKE_FAILURE` because the school firewall intercepted the connection.  
Fix: Used a VPN and `verify=False` in the code.

Official API:  
Hit a `ReadTimeoutError`. The official NBA server recognized the script and silently ignored the request until the 60-second timer ran out.

Third-Party API:  
Successfully connected through the firewall, but hit a `401 Unauthorized` error because the specific `stats` endpoint is now a paid feature.

Next steps / fix:  
Pivoting to a static dataset (CSV) from Kaggle. This eliminates all network/firewall "online" bottlenecks and allows the Surface Pro 9 to focus 100% of its compute power on machine learning rather than connection troubleshooting.
