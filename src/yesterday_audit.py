import pandas as pd
import joblib, os
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, leaguegamelog
from tabulate import tabulate

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "nba_v2_xgb_model.pkl")
STATS_PATH = os.path.join(ROOT, "data", "processed_matchups.csv")
LOG_PATH = os.path.join(ROOT, "data", "predictions_log.csv")

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


# Main function to audit yesterday's predictions against actual results
def run():
    yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    model = joblib.load(MODEL_PATH)
    df_stats = pd.read_csv(STATS_PATH)

    games = scoreboardv2.ScoreboardV2(game_date=yesterday).get_data_frames()[0]
    if games.empty:
        return print("No games yesterday.")

    # API Truth
    raw_log = leaguegamelog.LeagueGameLog(season="2025-26").get_data_frames()[0]
    outcome_map = dict(zip(raw_log["GAME_ID"].astype(str), raw_log["WL"]))

    table, hits = [], 0
    for _, g in games.iterrows():
        try:
            latest = df_stats[
                (df_stats["HOME_TEAM_ID"] == g["HOME_TEAM_ID"])
                | (df_stats["AWAY_TEAM_ID"] == g["HOME_TEAM_ID"])
            ].iloc[-1]
            prob = model.predict_proba(latest[FEATURES].values.reshape(1, -1))[0, 1]
            pick = "W" if prob > 0.5 else "L"

            actual = outcome_map.get(str(g["GAME_ID"]).zfill(10), "N/A")
            res = "✅" if pick == actual else "❌"
            if res == "✅":
                hits += 1
            home = g.get("HOME_TEAM_ABBREVIATION") or g.get("HOME_TEAM_CITY") or g["HOME_TEAM_ID"]
            away = g.get("VISITOR_TEAM_ABBREVIATION") or g.get("VISITOR_TEAM_CITY") or g["VISITOR_TEAM_ID"]
            table.append([home, away, pick, actual, res])
        except:
            continue

    print(f"\nAudit for {yesterday}:")
    print(
        tabulate(
            table, headers=["Home", "Away", "Pick", "Real", "Result"], tablefmt="psql"
        )
    )
    if table:
        print(f"Accuracy: {(hits/len(table))*100:.1f}%")


if __name__ == "__main__":
    run()
