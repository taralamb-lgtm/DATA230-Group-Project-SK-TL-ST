from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "crimes_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if ("hour" not in df.columns or "day_of_week" not in df.columns) and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = df["date"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.day_name()

    required_cols = ["primary_type", "district", "hour", "day_of_week", "domestic", "arrest"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy().dropna()

    df["primary_type"] = df["primary_type"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()
    df["day_of_week"] = df["day_of_week"].astype(str).str.strip()

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)

    df["domestic"] = df["domestic"].astype(str).str.strip().str.lower()
    df["domestic"] = df["domestic"].replace({
        "true": "Yes",
        "false": "No",
        "1": "Yes",
        "0": "No",
        "yes": "Yes",
        "no": "No"
    })

    if str(df["arrest"].dtype) == "bool":
        df["arrest"] = df["arrest"].astype(int)
    else:
        df["arrest"] = df["arrest"].astype(str).str.strip().str.lower()
        df["arrest"] = df["arrest"].replace({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0
        })
        df["arrest"] = pd.to_numeric(df["arrest"], errors="coerce")
        df = df.dropna(subset=["arrest"])
        df["arrest"] = df["arrest"].astype(int)

    return df


def build_model():
    categorical_features = ["primary_type", "district", "day_of_week", "domestic"]
    numeric_features = ["hour"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )
    return model


def main():
    df = load_data()

    X = df[["primary_type", "district", "hour", "day_of_week", "domestic"]]
    y = df["arrest"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "test_rows": int(len(X_test)),
    }

    with open(OUTPUT_DIR / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"]
    )
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    })
    roc_df.to_csv(OUTPUT_DIR / "roc_curve.csv", index=False)

    rf_model = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    prediction_output = X_test.copy()
    prediction_output["arrest_actual"] = y_test.values
    prediction_output["arrest_predicted"] = y_pred
    prediction_output["arrest_probability"] = y_prob
    prediction_output.to_csv(OUTPUT_DIR / "prediction_outputs.csv", index=False)

    print("Done. Files saved to outputs/")


if __name__ == "__main__":
    main()