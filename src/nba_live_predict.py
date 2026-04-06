import pandas as pd
import joblib
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime, timedelta


# This script is designed to be run on game days to provide live predictions for the day's matchups.
def run_predictor():
    try:
        # Load the frozen brain and stats
        model = joblib.load("nba_model.pkl")
        stats_lookup = pd.read_csv("latest_team_stats.csv")
        stats_lookup["team_id"] = stats_lookup["team_id"].astype(int)

        # Try to find games
        board = scoreboard.ScoreBoard()
        games = board.get_dict()["scoreboard"]["games"]

        # If today is empty or finished, the API might be looking for April 6
        if not games:
            print("No live games found. Checking upcoming schedule...")

        results = []
        print(f"Checking {len(games)} potential matchups...")

        for g in games:
            h_id = int(g["homeTeam"]["teamId"])
            a_id = int(g["awayTeam"]["teamId"])
            h_name = g["homeTeam"]["teamName"]
            a_name = g["awayTeam"]["teamName"]

            # Match against your database
            if (
                h_id in stats_lookup["team_id"].values
                and a_id in stats_lookup["team_id"].values
            ):
                h_data = stats_lookup[stats_lookup["team_id"] == h_id].iloc[-1]
                a_data = stats_lookup[stats_lookup["team_id"] == a_id].iloc[-1]

                # Build the 'Differential' features exactly as the model expects
                live_features = pd.DataFrame(
                    [
                        {
                            "pts_diff_5g": h_data["pts_avg"] - a_data["pts_avg"],
                            "reb_diff_5g": h_data["reb_avg"] - a_data["reb_avg"],
                            "rest_diff": 0,  # Neutral rest for future predictions
                        }
                    ]
                )

                prob = model.predict_proba(live_features)[0][1]

                results.append(
                    {
                        "Matchup": f"{a_name} @ {h_name}",
                        "Certainty": f"{prob*100:.1f}%",
                        "Pick": h_name if prob > 0.5 else a_name,
                        "Verdict": "BET" if (prob > 0.64 or prob < 0.36) else "PASS",
                    }
                )
            else:
                print(
                    f"Skipping {a_name} @ {h_name}: Team IDs missing from local database."
                )

        # Output the Table
        df_final = pd.DataFrame(results)
        print("\n--- LIVE NBA PREDICTOR ---")
        if not df_final.empty:
            # Sort by certainty to see the 'best' bets first
            df_final["Sort"] = df_final["Certainty"].str.replace("%", "").astype(float)
            df_final = df_final.sort_values("Sort", ascending=False).drop(
                columns=["Sort"]
            )
            print(df_final.to_string(index=False))
        else:
            print("No active games found. The NBA slate is currently empty.")

    except Exception as e:
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    run_predictor()
