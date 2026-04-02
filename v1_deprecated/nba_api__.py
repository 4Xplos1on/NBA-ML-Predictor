import pandas as pd
from nba_api.stats.endpoints import playergamelog

print("Testing official NBA API through VPN...")

custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com/'
}

try:
    log = playergamelog.PlayerGameLog(
        player_id='2544', 
        season='2023-24',
        headers=custom_headers,
        timeout=30
    )
    
    df = log.get_data_frames()[0]
    print(f"\nSuccess! Downloaded {len(df)} games.")
    print(df[['MATCHUP', 'WL', 'PTS']].head())

except Exception as e:
    print(f"\nConnection failed. Error: {e}")