import json
import sqlite3
import time

from src.database import (
    DB_PATH,
    create_database,
    insert_features,
    insert_label,
    insert_prediction,
    insert_transaction,
    to_utc_iso,
    utc_now_iso,
)
from src.feature_engineering import calculate_features
from src.fraud_predictor import load_model, predict_fraud
from src.transaction_generator import (
    EVENT_INTERVAL_SECONDS,
    HISTORY_PER_CUSTOMER,
    TransactionSimulator,
)
from src.validator import validate_transaction


# ----------------------------------------
# 1. QUARANTINE INVALID TRANSACTIONS
# ----------------------------------------

def quarantine(conn, transaction, reasons):
    transaction_id = (
        transaction.get("transaction_id", "")
        if isinstance(transaction, dict)
        else ""
    )
    conn.execute("""
        INSERT INTO quarantine_transactions
        (transaction_id, raw_data, rejection_reason, rejected_at)
        VALUES (?, ?, ?, ?)
    """, (
        str(transaction_id),
        json.dumps(transaction, default=str),
        "; ".join(reasons),
        utc_now_iso(),
    ))


# ----------------------------------------
# 2. PROCESS EACH TRANSACTION
# ----------------------------------------

def _store(conn, transaction, is_fraud):
    """STEP 3: Generate features, predict and store atomically.
    Returns (status, prediction)."""

    try:
        conn.execute("SAVEPOINT insert_transaction")

        features = calculate_features(conn, transaction)
        prediction = predict_fraud(features)

        insert_transaction(conn, transaction)
        insert_features(conn, features)
        insert_prediction(conn, transaction["transaction_id"], prediction)
        if is_fraud is not None:
            insert_label(conn, transaction["transaction_id"], is_fraud)

        conn.execute("RELEASE SAVEPOINT insert_transaction")
        return "ACCEPTED", prediction

    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO SAVEPOINT insert_transaction")
        conn.execute("RELEASE SAVEPOINT insert_transaction")

        # Only classify confirmed transaction ID collisions as duplicates.
        existing = conn.execute(
            "SELECT 1 FROM transactions WHERE transaction_id = ?",
            (transaction["transaction_id"],),
        ).fetchone()
        if existing is None:
            raise

        quarantine(conn, transaction, ["Duplicate transaction ID"])
        return "DUPLICATE", None


def process_transaction(transaction, db_path=DB_PATH, is_fraud=None):
    """Validate, featurise, score and store one transaction.

    `is_fraud` is the simulator's ground-truth label. It is stored
    separately so live model quality can be measured; it is never
    shown to the model.

    Returns "ACCEPTED", "REJECTED" or "DUPLICATE".
    """

    # STEP 1: Validate incoming transaction
    is_valid, errors = validate_transaction(transaction)

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        with conn:  # commits on success, rolls back on error
            if not is_valid:
                # STEP 2: Quarantine invalid data
                quarantine(conn, transaction, errors)
                status = "REJECTED"
            else:
                # Store timestamps in one fixed UTC format so they can be
                # compared and indexed as text.
                transaction = {
                    **transaction,
                    "timestamp": to_utc_iso(transaction["timestamp"]),
                }
                status, prediction = _store(conn, transaction, is_fraud)
    finally:
        conn.close()

    if status == "REJECTED":
        txn_id = (
            transaction.get("transaction_id")
            if isinstance(transaction, dict)
            else None
        )
        print(f"REJECTED | Transaction: {txn_id} | Reason: {errors}", flush=True)
        return status

    if status == "DUPLICATE":
        print(
            f"DUPLICATE | Transaction: {transaction['transaction_id']}",
            flush=True,
        )
        return status

    # STEP 4: Display successful prediction
    status = "SUSPICIOUS" if prediction["is_suspicious"] == 1 else "NORMAL"
    print(
        f"{status} | "
        f"Customer: {transaction['customer_id']} | "
        f"Amount: ₹{transaction['amount']:.2f} | "
        f"Fraud Score: {prediction['fraud_score']:.3f} | "
        f"Model: {prediction['model_version']}",
        flush=True,
    )
    return "ACCEPTED"


# ----------------------------------------
# 3. CUSTOMER HISTORY WARM-UP
# ----------------------------------------

# Share of live events that are malformed or duplicate records.
BAD_DATA_RATE = 0.04


def warm_up_history(simulator, db_path=DB_PATH):
    """Give customers back-history before the live stream starts.

    Without it most early transactions are a customer's first, so
    amount_ratio / city_changed carry no signal and fraud goes
    unflagged. The training replay applies the same warm-up, so the
    model sees the same distribution. Returns rows inserted.
    """

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        counts = dict(conn.execute(
            "SELECT customer_id, COUNT(*) FROM transactions GROUP BY customer_id"
        ).fetchall())
        needy = {
            c["customer_id"]
            for c in simulator.customers
            if counts.get(c["customer_id"], 0) < HISTORY_PER_CUSTOMER
        }
        if not needy:
            return 0

        rows = [t for t in simulator.history() if t["customer_id"] in needy]
        with conn:
            for transaction in rows:
                insert_transaction(conn, transaction)
        return len(rows)
    finally:
        conn.close()


# ----------------------------------------
# 4. START LIVE INGESTION SERVICE
# ----------------------------------------

def main():
    create_database()

    # Ensure model exists before accepting events.
    load_model()

    print("=" * 65)
    print("BANKING FRAUD DETECTION - LIVE MLOPS PIPELINE")
    print("=" * 65)
    print("Transaction Generator : ACTIVE")
    print("Data Validation       : ACTIVE")
    print("Feature Engineering   : ACTIVE")
    print("Fraud Prediction      : ACTIVE")
    print("SQLite Storage        : ACTIVE")
    print("Dashboard             : Run separately")
    print("-" * 65)

    simulator = TransactionSimulator(bad_data_rate=BAD_DATA_RATE)

    added = warm_up_history(simulator)
    if added:
        print(f"Customer history     : seeded {added:,} past transactions")

    try:
        while True:
            transaction, is_fraud = simulator.next()
            process_transaction(transaction, is_fraud=is_fraud)
            time.sleep(EVENT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nIngestion service stopped by user.")


if __name__ == "__main__":
    main()
