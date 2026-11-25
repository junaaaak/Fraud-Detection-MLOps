"""
Train the fraud model and tune its decision threshold.

Data is split by time (train -> validation -> test), as in production
the model always scores transactions that come after its training data.
The threshold is chosen on the validation set to maximise F1, then
final metrics are reported once on the untouched test set.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, MODELS_DIR, TRAINING_CSV

FEATURES = [
    "amount",
    "txn_count_5m",
    "amount_sum_1h",
    "amount_ratio",
    "city_changed",
]
MODEL_VERSION = "v2.1"


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def main():
    df = pd.read_csv(TRAINING_CSV).sort_values("timestamp")

    n = len(df)
    train = df.iloc[: int(n * 0.6)]
    valid = df.iloc[int(n * 0.6): int(n * 0.8)]
    test = df.iloc[int(n * 0.8):]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(train[FEATURES], train["fraud_label"])

    # Tune threshold on validation data.
    valid_scores = model.predict_proba(valid[FEATURES])[:, 1]
    threshold, valid_f1 = best_f1_threshold(valid["fraud_label"], valid_scores)
    print(f"Chosen threshold: {threshold:.3f} (validation F1 {valid_f1:.3f})")

    # Final, untouched evaluation on test data.
    test_scores = model.predict_proba(test[FEATURES])[:, 1]
    test_pred = (test_scores >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(test["fraud_label"], test_scores), 4),
        "pr_auc": round(average_precision_score(test["fraud_label"], test_scores), 4),
        "test_fraud_rate": round(float(test["fraud_label"].mean()), 4),
        "test_flag_rate": round(float(test_pred.mean()), 4),
    }

    print("\nTest set classification report:")
    print(classification_report(test["fraud_label"], test_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(test["fraud_label"], test_pred))
    print("Metrics:", metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "features": FEATURES,
        "version": MODEL_VERSION,
        "threshold": threshold,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    print(f"\nModel {MODEL_VERSION} saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
