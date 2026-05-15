import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

ML_FEATURES = [
    "srtt_ms", "rttvar_ms", "attempt",
    "success_rate_window", "inter_arrival_ms",
]

# Per-attempt RTO caps matching CoAP RFC 7252 backoff schedule
ATTEMPT_CAPS = {
    0: 2000.0,
    1: 4000.0,
    2: 8000.0,
    3: 16000.0,
    4: 32000.0,
    5: 60000.0,
    6: 60000.0,
}


def train_linear_model():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/training_data.csv").dropna()
    print(f"Loaded {len(df)} samples (raw)")

    if "is_timeout" in df.columns:
        before = len(df)
        df = df[df["is_timeout"] == 0]
        print(f"Filtered timeouts: {before} -> {len(df)} samples")

    df = df[(df["target_rto_ms"] >= 100) & (df["target_rto_ms"] <= 60000)]
    print(f"After RTO range filter: {len(df)} samples")

    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing features: {missing}")

    models      = {}
    all_pred_ms = []
    all_true_ms = []

    param_grid = {
        "svr__C":       [0.01, 0.1, 1.0, 10.0],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    }

    for att in sorted(df["attempt"].unique()):
        att = int(att)
        cap = ATTEMPT_CAPS.get(att, 60000.0)

        sub = df[df["attempt"] == att].copy()
        sub = sub[sub["target_rto_ms"] <= cap]

        if len(sub) < 30:
            print(f"  attempt={att}: only {len(sub)} samples — skipping")
            continue

        print(f"\n  attempt={att}: {len(sub)} samples, cap={cap:.0f}ms")

        X     = sub[ML_FEATURES].values
        y_log = np.log1p(np.clip(sub["target_rto_ms"].values, 100.0, cap))

        if len(sub) < 100:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", LinearSVR(C=1.0, epsilon=0.1,
                                  max_iter=1_000_000, tol=1e-3,
                                  dual="auto", random_state=42)),
            ])
            pipeline.fit(X, y_log)
            best = pipeline
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_log, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", LinearSVR(max_iter=1_000_000, tol=1e-3,
                                  dual="auto", random_state=42)),
            ])
            grid = GridSearchCV(pipeline, param_grid,
                                cv=3, scoring="r2", n_jobs=-1)
            grid.fit(X_tr, y_tr)
            best = grid.best_estimator_

            y_pred_log = np.clip(best.predict(X_te), 0.0, np.log1p(cap))
            y_pred_ms  = np.expm1(y_pred_log)
            y_true_ms  = np.expm1(y_te)

            r2  = r2_score(y_te, y_pred_log)
            mae = mean_absolute_error(y_true_ms, y_pred_ms)
            print(f"    R²(log)={r2:.4f}  MAE={mae:.1f}ms  "
                  f"params={grid.best_params_}")

            all_pred_ms.extend(y_pred_ms.tolist())
            all_true_ms.extend(y_true_ms.tolist())

        models[att] = best

    print(f"\n--- PER-ATTEMPT ENSEMBLE TRAINED ({len(models)} models) ---")
    if all_pred_ms:
        print(f"Overall R² (ms): {r2_score(all_true_ms, all_pred_ms):.4f}")
        print(f"Overall MAE:     {mean_absolute_error(all_true_ms, all_pred_ms):.2f} ms")

    print("\n--- SANITY CHECKS ---")
    for att, srtt, rttvar, sr in [
        (0, 200.0, 100.0, 1.0),
        (1, 300.0, 150.0, 0.8),
        (2, 500.0, 250.0, 0.5),
        (3, 800.0, 400.0, 0.2),
    ]:
        if att not in models:
            continue
        cap    = ATTEMPT_CAPS.get(att, 60000.0)
        sample = np.array([[srtt, rttvar, float(att), sr, 1200.0]])
        raw    = float(np.clip(models[att].predict(sample)[0],
                               0.0, np.log1p(cap)))
        pred   = np.expm1(raw)
        print(f"  attempt={att}: {pred:.0f}ms  (cap={cap:.0f}ms)")

    joblib.dump(models, "models/linear_rto_model.joblib")
    print("\nSaved -> models/linear_rto_model.joblib")


if __name__ == "__main__":
    train_linear_model()
