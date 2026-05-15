import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

ML_FEATURES = [
    "srtt_ms", "rttvar_ms", "attempt",
    "success_rate_window", "inter_arrival_ms",
]


def train_model():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/training_data.csv").dropna()
    print(f"Loaded {len(df)} samples (raw)")

    if "is_timeout" in df.columns:
        before = len(df)
        df = df[df["is_timeout"] == 0]
        print(f"Filtered timeouts: {before} -> {len(df)} samples")
    else:
        print("WARNING: no `is_timeout` column — re-run dataset_gen.py")

    df = df[(df["target_rto_ms"] >= 100) & (df["target_rto_ms"] <= 60000)]
    print(f"After RTO range filter: {len(df)} samples")

    if len(df) > 30_000:
        df = df.sample(n=30_000, random_state=42)
        print(f"Subsampled to {len(df)} for kernel SVR")

    if len(df) < 50:
        raise SystemExit("Not enough samples after filtering — check dataset_gen.py output.")

    stds = df[ML_FEATURES].std()
    low_var = [f for f in ML_FEATURES if stds[f] < 1e-9]
    if low_var:
        print(f"WARNING: near-zero variance features, adding tiny noise: {low_var}")
        for f in low_var:
            df[f] = df[f] + np.random.normal(0, 1e-6, size=len(df))

    X     = df[ML_FEATURES]
    y_log = np.log1p(df["target_rto_ms"].values)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_ms = np.expm1(y_test_log)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVR(kernel="rbf")),
    ])

    param_grid = {
        "svm__C":       [1.0, 10.0, 100.0],
        "svm__gamma":   ["scale", 0.1, 0.01],
        "svm__epsilon": [0.05, 0.1, 0.2],
    }

    print("Running GridSearchCV (RBF SVR)...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train_log)

    best = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")

    y_pred_log = best.predict(X_test)
    y_pred_ms  = np.expm1(np.clip(y_pred_log, 0.0, 12.0))

    print("\n--- KERNEL (RBF) SVR PERFORMANCE ---")
    print(f"Train R² (log): {best.score(X_train, y_train_log):.4f}")
    print(f"Test  R² (log): {r2_score(y_test_log, y_pred_log):.4f}")
    print(f"Test  R² (ms):  {r2_score(y_test_ms,  y_pred_ms):.4f}")
    print(f"Test  MAE:      {mean_absolute_error(y_test_ms, y_pred_ms):.2f} ms")

    sample = np.array([[200.0, 100.0, 0.0, 1.0, 1200.0]])
    raw    = best.predict(sample)[0]
    print(f"\nSanity check — raw log pred: {raw:.4f} → {np.expm1(raw):.1f} ms")

    joblib.dump(best, "models/rto_model.joblib")
    print("Saved -> models/rto_model.joblib")


if __name__ == "__main__":
    train_model()
