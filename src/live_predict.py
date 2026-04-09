import pandas as pd
import joblib, os
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv3
from tabulate import tabulate

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "nba_v2_xgb_model.pkl")
STATS_PATH = os.path.join(ROOT, "data", "processed_matchups.csv")
LOG_PATH = os.path.join(ROOT, "data", "predictions_log.csv")

THRESHOLD = 0.65

TEAM_MAP = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

FEATURES = [
    "PTS_DIFF",
    "REB_DIFF",
    "AST_DIFF",
    "TOV_DIFF",
    "FG_PCT_DIFF",
    "FG3_PCT_DIFF",
    "PLUS_MINUS_DIFF",
    "STL_DIFF",
    "BLK_DIFF",
    "eFG_PCT_DIFF",
    "PTS_10G_DIFF",
    "REB_10G_DIFF",
    "AST_10G_DIFF",
    "TOV_10G_DIFF",
    "FG_PCT_10G_DIFF",
    "FG3_PCT_10G_DIFF",
    "PLUS_MINUS_10G_DIFF",
    "STL_10G_DIFF",
    "BLK_10G_DIFF",
    "eFG_PCT_10G_DIFF",
    "REST_DAYS_DIFF",
    "MISSING_PTS_DIFF",
    "B2B_DIFF",
    "STREAK_DIFF",
    "PTS_10G_STD_DIFF",
    "ALTITUDE_FLAG",
    "ELO_DIFF",
]


def run():
    if not os.path.exists(MODEL_PATH):
        return print("Model missing.")

    df_stats = pd.read_csv(STATS_PATH)
    model = joblib.load(MODEL_PATH)
    today = datetime.now().strftime("%Y-%m-%d")

    # Diagnostic: Check the latest date in your data
    latest_data_date = df_stats["GAME_DATE"].max()
    print(f"Engine Status: Data current through {latest_data_date}")

    try:
        sb = scoreboardv3.ScoreboardV3(game_date=today).get_dict()
        games = sb["scoreboard"]["games"]
    except:
        return print("API Connection Error.")

    if not games:
        return print(f"No games found for today ({today}).")

    results, log_data = [], []
    for g in games:
        try:
            h_id, a_id = g["homeTeam"]["teamId"], g["awayTeam"]["teamId"]
            h_name, a_name = TEAM_MAP.get(h_id, h_id), TEAM_MAP.get(a_id, a_id)

            # Pull the most recent matchup data for these teams
            team_stats = df_stats[
                (df_stats["HOME_TEAM_ID"] == h_id) | (df_stats["AWAY_TEAM_ID"] == h_id)
            ]

            if team_stats.empty:
                print(f"⚠️ {h_name} vs {a_name}: No stats found in CSV.")
                continue

            latest = team_stats.iloc[-1]
            prob = model.predict_proba(latest[FEATURES].values.reshape(1, -1))[0, 1]

            pick = h_name if prob > 0.5 else a_name
            conf_val = prob if prob > 0.5 else (1 - prob)
            verdict = "BET" if conf_val >= THRESHOLD else "PASS"
            conf_str = f"{conf_val*100:.1f}%"

            results.append([h_name, a_name, pick, conf_str, verdict])
            log_data.append(
                {
                    "GAME_ID": g["gameId"],
                    "DATE": today,
                    "HOME": h_name,
                    "AWAY": a_name,
                    "PREDICTION": pick,
                    "CERTAINTY": conf_str,
                    "VERDICT": verdict,
                    "ACTUAL_RESULT": "PENDING",
                }
            )
        except:
            continue

    # Display results to terminal
    if results:
        print(f"\nDakota Engine: {today}")
        print(
            tabulate(
                results,
                headers=["Home", "Away", "Pick", "Conf", "Verdict"],
                tablefmt="psql",
            )
        )

        # --- REFINED LOGGING WITH DUPLICATE CHECK ---
        new_df = pd.DataFrame(log_data)
        if os.path.exists(LOG_PATH):
            try:
                old_df = pd.read_csv(LOG_PATH)
                # Filter out any GAME_IDs that already exist in the log
                new_df = new_df[
                    ~new_df["GAME_ID"].astype(str).isin(old_df["GAME_ID"].astype(str))
                ]

                if not new_df.empty:
                    new_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
            except:
                # Fallback: create fresh if read fails
                new_df.to_csv(LOG_PATH, index=False)
        else:
            new_df.to_csv(LOG_PATH, index=False)
    else:
        print(
            "\nAll games skipped. Ensure your Pipeline has run for the 2025-26 season."
        )


if __name__ == "__main__":
    run()
