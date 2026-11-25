"""
Build the training set by replaying the live pipeline offline.

Transactions come from the same simulator the ingestion service uses,
and features come from the same `calculate_features` function, run
against an in-memory SQLite database. This keeps training and serving
features identical (no train/serve skew).
"""

import random
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import DATA_DIR, TRAINING_CSV
from src.database import create_schema, insert_transaction
from src.feature_engineering import calculate_features
from src.transaction_generator import EVENT_INTERVAL_SECONDS, TransactionSimulator

N_TRANSACTIONS = 30_000
SEED = 42


def main():
    simulator = TransactionSimulator(seed=SEED)
    clock_rng = random.Random(SEED)

    conn = sqlite3.connect(":memory:")
    create_schema(conn)

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Same customer back-history the live pipeline seeds at startup.
    for transaction in simulator.history(end=now):
        insert_transaction(conn, transaction)

    rows = []

    for _ in range(N_TRANSACTIONS):
        # Same average spacing as the live stream, with some jitter.
        now += timedelta(
            seconds=clock_rng.uniform(
                0.5 * EVENT_INTERVAL_SECONDS, 1.5 * EVENT_INTERVAL_SECONDS
            )
        )
        transaction, is_fraud = simulator.next(now=now)

        features = calculate_features(conn, transaction)
        insert_transaction(conn, transaction)

        rows.append({
            "transaction_id": transaction["transaction_id"],
            "customer_id": transaction["customer_id"],
            "timestamp": transaction["timestamp"],
            "amount": features["amount"],
            "txn_count_5m": features["txn_count_5m"],
            "amount_sum_1h": features["amount_sum_1h"],
            "amount_ratio": features["amount_ratio"],
            "city_changed": features["city_changed"],
            "fraud_label": is_fraud,
        })

    conn.close()

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_CSV, index=False)

    print(f"Training data saved to '{TRAINING_CSV}'")
    print("Total transactions:", len(df))
    print(f"Fraudulent: {df['fraud_label'].sum()} ({df['fraud_label'].mean():.1%})")
    print(df.describe().round(2).T)


if __name__ == "__main__":
    main()
