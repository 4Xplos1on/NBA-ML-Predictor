import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta # Import to target TODAY
import json

# Get today's date, and subtract one day to have up-to-date data
today = datetime.now()
yesterday = datetime.now() - timedelta(days=1)

# Getting to the site and accesing data for TODAY
# The URL formated so it takes in today's date
url = f'https://www.basketball-reference.com/boxscores/?month={yesterday.month}&day={yesterday.day}&year={yesterday.year}'

# Create a disguise for the scraper
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Pass the headers into the get request
response = requests.get(url, headers=headers)
# Check for us reaching the site
print(f"Status Code: {response.status_code}")
print(f"Checking URL: {url}")

soup = BeautifulSoup(response.content, 'html.parser')

# Initialize list to store all games
nba_games = []

# Find all the game boxes on the page
game_summaries = soup.find_all('div', class_='game_summary')

# Print availible games
print(f"Found {len(game_summaries)} games.")

for game in game_summaries:
    # Extracting the data
    # 1. Find all the table rows in this game box
    rows = game.find_all('tr')

    # 2. Get the 2 teams
    away_row = rows[0]
    home_row = rows[1]

    # 3. Extract the names
    away_name = away_row.find('a').text
    homw_name = home_row.find('a').text

    # 4. Extract the scores
    away_score = int(away_row.find('td', class_='right').text)
    home_score = int(home_row.find('td', class_='right').text)

    # ictionary for this specific game
    game_data = {
        "date": yesterday.strftime("%Y-%m-%d"),
        "away_team": away_name,
        "away_score": away_score,
        "home_team": home_name,
        "home_score": home_score
    }
    
    # Add it to nba_games
    nba_games.append(game_data)

###print(nba_games)
