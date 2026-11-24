import sqlite3
from datetime import datetime, timezone

from src.config import DB_PATH

SCHEMA = {
    "transactions": """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT,
            amount REAL,
            merchant TEXT,
            city TEXT,
            device TEXT,
            timestamp TEXT
        )
    """,
    # Rejected transactions, kept for data-quality monitoring.
    "quarantine_transactions": """
        CREATE TABLE IF NOT EXISTS quarantine_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            raw_data TEXT,
            rejection_reason TEXT NOT NULL,
            rejected_at TEXT NOT NULL
        )
    """,
    "transaction_features": """
        CREATE TABLE IF NOT EXISTS transaction_features (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            amount REAL NOT NULL,
            txn_count_5m INTEGER NOT NULL,
            amount_sum_1h REAL NOT NULL,
            avg_prior_amount REAL,
            amount_ratio REAL,
            city_changed INTEGER NOT NULL,
            feature_timestamp TEXT NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
        )
    """,
    "fraud_predictions": """
        CREATE TABLE IF NOT EXISTS fraud_predictions (
            transaction_id TEXT PRIMARY KEY,
            fraud_score REAL NOT NULL,
            is_suspicious INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            predicted_at TEXT NOT NULL,
            FOREIGN KEY (transaction_id)
                REFERENCES transactions(transaction_id)
        )
    """,
    # Ground truth from the simulator, so live model quality
    # (precision / recall) can be measured on the dashboard.
    "transaction_labels": """
        CREATE TABLE IF NOT EXISTS transaction_labels (
            transaction_id TEXT PRIMARY KEY,
            is_fraud INTEGER NOT NULL,
            FOREIGN KEY (transaction_id)
                REFERENCES transactions(transaction_id)
        )
    """,
}

INDEXES = [
    # Feature engineering looks up a customer's recent history.
    """CREATE INDEX IF NOT EXISTS idx_transactions_customer_time
       ON transactions (customer_id, timestamp)""",
    """CREATE INDEX IF NOT EXISTS idx_transactions_time
       ON transactions (timestamp)""",
    # The dashboard filters by time window.
    """CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at
       ON fraud_predictions (predicted_at)""",
    """CREATE INDEX IF NOT EXISTS idx_quarantine_rejected_at
       ON quarantine_transactions (rejected_at)""",
]


def utc_now_iso():
    return to_utc_iso(datetime.now(timezone.utc))


def to_utc_iso(value):
    """Fixed-format UTC timestamp, so stored timestamps can be compared
    and indexed as plain strings."""

    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


class SchemaMismatchError(RuntimeError):
    pass


def _check_schema(conn):
    """Fail loudly if an existing table is missing expected columns,
    instead of silently dropping data."""

    for table, ddl in SCHEMA.items():
        reference = sqlite3.connect(":memory:")
        reference.execute(ddl)
        expected = {
            row[1] for row in reference.execute(f"PRAGMA table_info({table})")
        }
        reference.close()

        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        missing = expected - actual
        if missing:
            raise SchemaMismatchError(
                f"Table '{table}' is missing columns {sorted(missing)}. "
                f"Migrate it, or back up and delete the database file to "
                f"recreate it."
            )


def create_schema(conn):
    for ddl in SCHEMA.values():
        conn.execute(ddl)
    _check_schema(conn)
    for ddl in INDEXES:
        conn.execute(ddl)


def create_database(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # WAL lets the dashboard read while ingestion is writing.
        conn.execute("PRAGMA journal_mode=WAL")
        create_schema(conn)
    conn.close()

    print("Database is ready at:", db_path)


# ----------------------------------------
# WRITES (shared by ingestion and training-data generation)
# ----------------------------------------

def insert_transaction(conn, transaction):
    conn.execute("""
        INSERT INTO transactions
        (transaction_id, customer_id, amount, merchant, city, device, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        transaction["transaction_id"],
        transaction["customer_id"],
        transaction["amount"],
        transaction["merchant"],
        transaction["city"],
        transaction["device"],
        transaction["timestamp"],
    ))


def insert_features(conn, features):
    conn.execute("""
        INSERT INTO transaction_features
        (transaction_id, customer_id, amount, txn_count_5m, amount_sum_1h,
         avg_prior_amount, amount_ratio, city_changed, feature_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        features["transaction_id"],
        features["customer_id"],
        features["amount"],
        features["txn_count_5m"],
        features["amount_sum_1h"],
        features["avg_prior_amount"],
        features["amount_ratio"],
        features["city_changed"],
        features["feature_timestamp"],
    ))


def insert_prediction(conn, transaction_id, prediction):
    conn.execute("""
        INSERT INTO fraud_predictions
        (transaction_id, fraud_score, is_suspicious, model_version, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        transaction_id,
        prediction["fraud_score"],
        prediction["is_suspicious"],
        prediction["model_version"],
        utc_now_iso(),
    ))


def insert_label(conn, transaction_id, is_fraud):
    conn.execute("""
        INSERT OR REPLACE INTO transaction_labels (transaction_id, is_fraud)
        VALUES (?, ?)
    """, (transaction_id, int(is_fraud)))


if __name__ == "__main__":
    create_database()
