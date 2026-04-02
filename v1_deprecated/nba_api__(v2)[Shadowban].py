import pandas as pd
from nba_api.stats.endpoints import playergamelog

# Test NBA API connection
print("Testing official NBA API through VPN")

# Headers to help request go through
custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com/'
}

try:
    # Get player game data
    log = playergamelog.PlayerGameLog(
        player_id='2544',
        season='2023-24',
        headers=custom_headers,
        timeout=30
    )
    
    # Convert to dataframe
    df = log.get_data_frames()[0]
    
    # Show result
    print("Success")
    print(df[['MATCHUP', 'WL', 'PTS']].head())

except Exception as e:
    # Show error if it fails
    print("Connection failed")
    print(e)
