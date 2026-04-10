import os, subprocess, time
from tabulate import tabulate


def clear():
    os.system("cls" if os.name == "nt" else "clear")


# Run a subprocess and return success status
def run(name, path):
    print(f">> {name}...")
    try:
        subprocess.run(["python", path], check=True, capture_output=False)
        return True
    except:
        print(f"[!] {name} Failed.")
        return False


# Audit yesterday's predictions against actual results
def main():
    while True:
        clear()
        print("=" * 40 + "\n   NBA PREDICTOR 2026 - HUB\n" + "=" * 40)

        menu = [
            ["1", "PIPELINE", "Sync Data & Re-train"],
            ["2", "AUDIT", "Check Yesterday's Picks"],
            ["3", "PREDICT", "Analyze Tonight's Games"],
            ["4", "EXIT", "Close Engine"],
        ]
        print(tabulate(menu, headers=["#", "Mode", "Goal"], tablefmt="simple"))

        choice = input("\nSelect (1-4): ")

        if choice == "1":
            clear()
            if run("Data", "src/nba_api-datareq.py"):
                if run("Process", "src/processor.py"):
                    run("Train", "src/nba-predict_v2.py")
            input("\nDone. Press Enter...")
        elif choice == "2":
            clear()
            run("Audit", "src/yesterday_audit.py")
            input("\nDone. Press Enter...")
        elif choice == "3":
            clear()
            run("Predict", "src/live_predict.py")
            input("\nDone. Press Enter...")
        elif choice == "4":
            break


if __name__ == "__main__":
    main()
