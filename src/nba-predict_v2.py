import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import joblib  # To save the model
import os

# Paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_path = os.path.join(base_path, "data", "processed_matchups.csv")
model_save_path = os.path.join(base_path, "models", "nba_v2_xgb_model.pkl")

features = [
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
target = "TARGET"


# Split the data into train and test, not using train_test_split to maintain the order of the games
def split_data(path):
    df = pd.read_csv(path)
    # Convert GAME_DATE to datetime for proper sorting
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Use the first 80% of the data for training and the last 20% for testing
    index = int(len(df) * 0.8)
    train = df.iloc[:index]
    test = df.iloc[index:]

    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]

    # Diagnose class imbalance, that might be causing the model to predict wins
    print("Class distribution (Training - Raw Count):")
    print(y_train.value_counts())
    print("\nClass distribution (Training - Percentages):")
    print(y_train.value_counts(normalize=True).round(3))

    print(
        f"\nSplit complete: {len(train)} Training samples | {len(test)} Testing samples."
    )

    return x_train, y_train, x_test, y_test


# Train the XGBoost model
def train_model(x_train, y_train):

    # Fix the bias in XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    # Tune XGBoost hyperparameters using GridSearchCV
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
    }

    print("Running Grid Search to find optimal parameters...")
    grid = GridSearchCV(
        xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)

    print("\n Model Trained")

    model = grid.best_estimator_

    # See what the model is actually using, to diagnose if it's just picking up on a few features
    feat_imp = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nImportant features:")
    print("-" * 40)
    print(feat_imp.head(10).to_string(index=False))

    return model


# Evaluate the model using cross-validation and on the test set
def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    # Extract the raw probability decimal (e.g., 0.82)
    probabilities = model.predict_proba(x_test)[:, 1]

    print("Classification Report:")
    print(f"Accuracy Score: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")

    # Display a quick sample of Certainty Scores from the test set
    print("\nSample Certainty Scores (First 5 Test Games):")
    for i in range(5):
        certainty = probabilities[i] * 100
        prediction = "Win" if predictions[i] == 1 else "Loss"
        print(f"Game {i+1}: Predicted {prediction} with {certainty:.1f}% certainty.")

    print("=" * 50 + "\n")


# Save the model using joblib
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved to: {path}\n")


# Main
def main():
    x_train, y_train, x_test, y_test = split_data(processed_data_path)
    model = train_model(x_train, y_train)
    evaluate(model, x_test, y_test)
    save_model(model, model_save_path)


if __name__ == "__main__":
    main()
