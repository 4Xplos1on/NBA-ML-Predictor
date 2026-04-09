from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import time
import os

# Set up paths to save directly to your data folder
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_path, "data")
os.makedirs(data_dir, exist_ok=True)


def download_teams():
    print("Downloading Team Data...")
    print("\n----------------------------------------")
    seasons = [
        "2023-24",
        "2024-25",
    ]  # Kept to 2 years for optimal Surface Pro performance
    all_seasons_data = []

    for season in seasons:
        try:
            print(f"Getting Team data for {season}...")
            log = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season"
            )
            df = log.get_data_frames()[0]
            all_seasons_data.append(df)
            time.sleep(2)

        except Exception as e:
            print(f"Error downloading Team data for {season}: {e}")

    return all_seasons_data


def download_players():
    print("Downloading Player Data...")
    seasons = ["2023-24", "2024-25"]
    all_players_data = []

    for season in seasons:
        try:
            print(f"Getting Player data for {season}...")
            # 'P' tells the API to get individual player stats instead of team stats
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
            )
            df = log.get_data_frames()[0]
            all_players_data.append(df)
            time.sleep(2)

        except Exception as e:
            print(f"Error downloading Player data for {season}: {e}")

    return all_players_data


if __name__ == "__main__":
    # Download and save Team Data
    team_list = download_teams()
    if team_list:
        full_team_data = pd.concat(team_list, ignore_index=True)
        team_csv_path = os.path.join(data_dir, "nba_raw_2yr.csv")
        full_team_data.to_csv(team_csv_path, index=False)
        print(f"Team data saved! ({len(full_team_data)} rows)")

    # Download and save Player Data
    player_list = download_players()
    if player_list:
        full_player_data = pd.concat(player_list, ignore_index=True)
        player_csv_path = os.path.join(data_dir, "nba_players_raw.csv")
        full_player_data.to_csv(player_csv_path, index=False)
        print(f"Player data saved! ({len(full_player_data)} rows)")
