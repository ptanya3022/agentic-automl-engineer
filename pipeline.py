import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import optuna

# ------------------ Data Preprocessing ------------------
def preprocess(df, target_column):
    df = df.dropna()  # remove rows with missing values
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------ Objective Function ------------------
def objective(trial, X_train, y_train, X_valid, y_valid):
    model_name = trial.suggest_categorical("model", ["xgb", "rf", "lr"])

    if model_name == "xgb":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10)
        )
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)

# ------------------ Main Pipeline ------------------
def run_pipeline(df, target_column):
    X_train, X_valid, y_train, y_valid = preprocess(df, target_column)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=20)

    best_params = study.best_trial.params
    best_model_name = best_params.pop("model")

    if best_model_name == "xgb":
        best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="mlogloss")
    elif best_model_name == "rf":
        best_model = RandomForestClassifier(**best_params)
    else:
        best_model = LogisticRegression(max_iter=1000)

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)

    report = classification_report(y_valid, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return best_model, acc, report_df
