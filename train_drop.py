import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.calibration import CalibratedClassifierCV

DROP_FEATURES = [
    "srtt_ms", "rttvar_ms",
    "success_rate_window", "inter_arrival_ms",
]


def train_drop_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots",  exist_ok=True)

    df = pd.read_csv("data/training_data.csv").dropna()
    print(f"Loaded {len(df)} total samples")

    if "drop_label" not in df.columns:
        raise SystemExit("ERROR: no drop_label column — re-run dataset_gen.py")

    print("\n--- Drop label distribution ---")
    vc = df["drop_label"].value_counts()
    print(vc)
    pos_rate = vc.get(1, 0) / len(df) * 100
    print(f"Positive (timeout) rate: {pos_rate:.1f}%")

    if df["drop_label"].nunique() < 2:
        print(
            f"\nWARNING: drop_label contains only one class "
            f"(positive rate {pos_rate:.2f}%).\n"
            "Re-run data_from_sim.py to generate timeout samples.\n"
            "Saving a dummy drop model so downstream scripts don't crash."
        )
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="constant", constant=0)
        dummy.fit([[0, 0, 0, 0]], [0])
        joblib.dump(dummy, "models/drop_model.joblib")
        print("Saved dummy -> models/drop_model.joblib")
        return

    if pos_rate < 1.0:
        print(f"\nWARNING: very low positive rate ({pos_rate:.2f}%). Classifier may have poor recall.")

    missing = [f for f in DROP_FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing features: {missing}")

    X = df[DROP_FEATURES].values
    y = df["drop_label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Positive rate — Train: {y_train.mean()*100:.1f}%  Test: {y_test.mean()*100:.1f}%")

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    model = CalibratedClassifierCV(rf, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print("\n--- DROP CLASSIFIER PERFORMANCE ---")
    print(classification_report(y_test, y_pred,
                                 target_names=["Success (0)", "Timeout (1)"]))
    print(f"ROC-AUC: {auc:.4f}")

    print("\n--- FEATURE IMPORTANCE ---")
    base_rf = model.calibrated_classifiers_[0].estimator
    for feat, imp in sorted(zip(DROP_FEATURES, base_rf.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {feat:25s}: {imp:.4f}")

    print("\n--- THRESHOLD ANALYSIS ---")
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'FalseDropRate':>14}")
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
        yt = (y_prob >= thresh).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            y_test, yt, average=None, labels=[0,1], zero_division=0)
        fdr = ((yt==1)&(y_test==0)).sum() / max(1,(y_test==0).sum())
        print(f"{thresh:>10.2f}  {p[1]:>10.3f}  {r[1]:>8.3f}  {fdr:>14.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["Success","Timeout"]).plot(ax=ax)
    ax.set_title("Drop Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/drop_model_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/drop_model_confusion_matrix.png")

    joblib.dump(model, "models/drop_model.joblib")
    print("Saved -> models/drop_model.joblib")


if __name__ == "__main__":
    train_drop_model()
