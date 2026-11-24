from functools import lru_cache

import joblib
import pandas as pd

from src.config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one with "
            f"`python -m scripts.generate_training_data` then "
            f"`python -m scripts.train_model`."
        )
    return joblib.load(model_path)


def predict_fraud(features):
    artifact = load_model()
    model = artifact["model"]
    columns = artifact["features"]
    threshold = artifact["threshold"]

    # Missing values (e.g. amount_ratio for a customer's first
    # transaction) become NaN so the pipeline's imputer handles them.
    X = pd.DataFrame(
        [{name: features.get(name) for name in columns}],
        columns=columns,
        dtype=float,
    )

    probability = float(model.predict_proba(X)[0][1])

    return {
        "fraud_score": probability,
        "is_suspicious": int(probability >= threshold),
        "model_version": artifact["version"],
    }


if __name__ == "__main__":
    example = {
        "amount": 25000,
        "txn_count_5m": 3,
        "amount_sum_1h": 60000,
        "amount_ratio": 10,
        "city_changed": 1,
    }
    print("Prediction:", predict_fraud(example))
