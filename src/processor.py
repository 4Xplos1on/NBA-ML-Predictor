# Transforms the raw data into a format that can be used by the model
import pandas as pd
import os
import numpy as np

# Paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Works on Windows and Linux and MacOS
raw_data_path = os.path.join(base_path, "data", "nba_raw_2yr.csv")
processed_data_path = os.path.join(base_path, "data", "processed_matchups.csv")

# For Elo Ratings
rolling_stats = [
    "PTS",
    "REB",
    "AST",
    "TOV",
    "FG_PCT",
    "FG3_PCT",
    "PLUS_MINUS",
    "STL",
    "BLK",
    "eFG_PCT",
]
window = 5


# Main function to run all the steps in order
def process_data():
    df = pd.read_csv(raw_data_path)
    # Convert for pandas to recognize as date and to extract home/away information
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs\\.")

    # Calculate Effective Field Goal Percentage (eFG%)
    df["eFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]

    print(f"Processing data from {raw_data_path}...")
    print("-----------------------------------------")
    return df


# Add rest days and back-to-back flags for each team
def add_rest_days(df):
    df = df.sort_values("GAME_DATE")
    df["REST_DAYS"] = (
        df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(3)
    )  # Assuming 3 rest days for the first game of each team

    # Binary flag for Back-to-Back games
    df["IS_B2B"] = (df["REST_DAYS"] <= 1).astype(int)
    return df


def add_streaks(df):
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["WIN_VAL"] = (df["WL"] == "W").astype(int) * 2 - 1

    # Fix: Ensure streak_id is calculated per team
    df["streak_id"] = df.groupby("TEAM_ID")["WIN_VAL"].transform(
        lambda x: (x != x.shift(1)).cumsum()
    )

    df["STREAK"] = df.groupby(["TEAM_ID", "streak_id"]).cumcount() + 1
    df["STREAK"] = df["STREAK"] * df["WIN_VAL"]

    df["STREAK"] = df.groupby("TEAM_ID")["STREAK"].shift(1).fillna(0)
    # Clean up the helper column
    df = df.drop(columns=["streak_id"])
    return df


# Add rolling averages for the specified stats
def add_rolling_stats(df):
    df = df.sort_values("GAME_DATE")
    for stat in rolling_stats:
        df[f"{stat}_5G_AVG"] = df.groupby("TEAM_ID")[stat].transform(
            lambda x: x.shift(1).ewm(span=5).mean()
        )
        df[f"{stat}_10G_AVG"] = df.groupby("TEAM_ID")[stat].transform(
            lambda x: x.shift(1).ewm(span=10).mean()
        )

    # Consistency Feature - Standard Deviation of points over the last 10 games
    df["PTS_10G_STD"] = df.groupby("TEAM_ID")["PTS"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std().fillna(0)
    )
    return df


def build_matchups(df):
    home = df[df["IS_HOME"]].copy()
    away = df[~df["IS_HOME"]].copy()

    rolling_cols = [f"{stat}_5G_AVG" for stat in rolling_stats] + [
        f"{stat}_10G_AVG" for stat in rolling_stats
    ]

    # Added new features to the merge list
    keep_columns = [
        "GAME_ID",
        "TEAM_ABBREVIATION",
        "WL",
        "SEASON_ID",
        "TEAM_ID",
        "GAME_DATE",
        "REST_DAYS",
        "IS_B2B",
        "STREAK",
        "PTS_10G_STD",
    ] + rolling_cols

    home = home[keep_columns].rename(
        columns={
            c: f"HOME_{c}"
            for c in keep_columns
            if c not in ["GAME_ID", "GAME_DATE", "SEASON_ID"]
        }
    )
    away = away[keep_columns].rename(
        columns={
            c: f"AWAY_{c}"
            for c in keep_columns
            if c not in ["GAME_ID", "GAME_DATE", "SEASON_ID"]
        }
    )

    merged = pd.merge(home, away, on=["GAME_ID", "GAME_DATE", "SEASON_ID"])
    print(f"Built matchups with {len(merged)} rows.")
    return merged


# Compute the differences between the home and away rolling averages for each stat
def compute_differences(df):
    for stat in rolling_stats:
        df[f"{stat}_DIFF"] = df[f"HOME_{stat}_5G_AVG"] - df[f"AWAY_{stat}_5G_AVG"]
        df[f"{stat}_10G_DIFF"] = df[f"HOME_{stat}_10G_AVG"] - df[f"AWAY_{stat}_10G_AVG"]

    df["REST_DAYS_DIFF"] = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]
    df["B2B_DIFF"] = df["HOME_IS_B2B"] - df["AWAY_IS_B2B"]
    df["STREAK_DIFF"] = df["HOME_STREAK"] - df["AWAY_STREAK"]
    df["PTS_10G_STD_DIFF"] = df["HOME_PTS_10G_STD"] - df["AWAY_PTS_10G_STD"]

    # Altitude Flag
    df["ALTITUDE_FLAG"] = df["HOME_TEAM_ABBREVIATION"].isin(["DEN", "UTA"]).astype(int)

    return df


# Numbers into labels for the target variable (0, 1)
def add_target(df):
    df["TARGET"] = (df["HOME_WL"] == "W").astype(int)
    return df


def calculate_elo(df):
    print("Calculating Dynamic Elo Ratings...")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    elo_dict = {}
    home_elos, away_elos = [], []  # Fix: Use plural 'home_elos'

    for index, row in df.iterrows():
        home_team = row["HOME_TEAM_ID"]
        away_team = row["AWAY_TEAM_ID"]

        current_h_elo = elo_dict.get(home_team, 1500)
        current_a_elo = elo_dict.get(away_team, 1500)

        home_elos.append(current_h_elo)
        away_elos.append(current_a_elo)

        # Expected win probability
        expected_home = 1 / (1 + 10 ** ((current_a_elo - (current_h_elo + 65)) / 400))
        expected_away = 1 - expected_home

        home_win = row["TARGET"]

        # Update Elos factor 20
        elo_dict[home_team] = current_h_elo + 20 * (home_win - expected_home)
        elo_dict[away_team] = current_a_elo + 20 * ((1 - home_win) - expected_away)

    df["HOME_ELO"] = home_elos  # Fix: Assign the list, not the last integer
    df["AWAY_ELO"] = away_elos
    df["ELO_DIFF"] = df["HOME_ELO"] - df["AWAY_ELO"]
    return df


# Calculate missing points from impact players (injuries/rest)
def calculate_injuries(df):
    print("Calculating Missing Player Production...")
    players_path = os.path.join(base_path, "data", "nba_players_raw.csv")
    players = pd.read_csv(players_path)

    player_avg = (
        players.groupby(["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"])["PTS"]
        .mean()
        .reset_index()
    )
    player_avg = player_avg.rename(columns={"PTS": "AVG_PTS"})

    baseline_path = os.path.join(base_path, "data", "player_baselines.csv")
    player_avg.to_csv(baseline_path, index=False)

    impact_players = player_avg[player_avg["AVG_PTS"] >= 10.0]
    missing_data = []
    different_games = players["GAME_ID"].unique()

    for game_id in different_games:
        played_game = players[players["GAME_ID"] == game_id]["PLAYER_ID"].tolist()
        teams_in = players[players["GAME_ID"] == game_id]["TEAM_ID"].unique()

        for team in teams_in:
            team_impact_players = impact_players[impact_players["TEAM_ID"] == team]
            missing_pts = 0

            for _, row in team_impact_players.iterrows():
                if row["PLAYER_ID"] not in played_game:
                    missing_pts += row["AVG_PTS"]

            missing_data.append(
                {"GAME_ID": game_id, "TEAM_ID": team, "MISSING_PTS": missing_pts}
            )

    missing_df = pd.DataFrame(missing_data)

    df = pd.merge(
        df,
        missing_df.rename(
            columns={"TEAM_ID": "HOME_TEAM_ID", "MISSING_PTS": "HOME_MISSING_PTS"}
        ),
        on=["GAME_ID", "HOME_TEAM_ID"],
        how="left",
    )
    df = pd.merge(
        df,
        missing_df.rename(
            columns={"TEAM_ID": "AWAY_TEAM_ID", "MISSING_PTS": "AWAY_MISSING_PTS"}
        ),
        on=["GAME_ID", "AWAY_TEAM_ID"],
        how="left",
    )

    df["HOME_MISSING_PTS"] = df["HOME_MISSING_PTS"].fillna(0)
    df["AWAY_MISSING_PTS"] = df["AWAY_MISSING_PTS"].fillna(0)
    df["MISSING_PTS_DIFF"] = df["HOME_MISSING_PTS"] - df["AWAY_MISSING_PTS"]

    return df


# Clean the data by keeping only the relevant columns and dropping rows with NaN values
def clean_save(df, path):
    diff_5g = [f"{stat}_DIFF" for stat in rolling_stats]
    diff_10g = [f"{stat}_10G_DIFF" for stat in rolling_stats]

    # Packaged all new features into the final save block
    all_diffs = (
        diff_5g
        + diff_10g
        + [
            "REST_DAYS_DIFF",
            "MISSING_PTS_DIFF",
            "B2B_DIFF",
            "STREAK_DIFF",
            "PTS_10G_STD_DIFF",
            "ALTITUDE_FLAG",
            "ELO_DIFF",
        ]
    )

    final_columns = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_ID",
        "HOME_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ID",
        "AWAY_TEAM_ABBREVIATION",
        "TARGET",
    ] + all_diffs

    df_output = df[final_columns].dropna(subset=diff_5g)
    df_output.to_csv(path, index=False)

    print("----------------------------------------")
    print(f"Saved processed data to {path}")
    print(f"Total Rows: {len(df_output)}")
    print("----------------------------------------")
    return df_output


# Main function to run all the steps in order
def main():
    df = process_data()
    df = add_rest_days(df)
    df = add_streaks(df)
    df = add_rolling_stats(df)
    df = build_matchups(df)
    df = compute_differences(df)
    df = add_target(df)
    df = calculate_elo(df)
    df = calculate_injuries(df)
    df_final = clean_save(df, processed_data_path)
    return df_final


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
