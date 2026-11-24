import sqlite3

import pytest

from src import ingestion
from src.config import MODEL_PATH
from src.database import SchemaMismatchError, create_database


@pytest.fixture(autouse=True)
def fixed_prediction(monkeypatch):
    """Keep ingestion tests independent of the trained model."""

    monkeypatch.setattr(ingestion, "predict_fraud", lambda features: {
        "fraud_score": 0.9,
        "is_suspicious": 1,
        "model_version": "test",
    })


def count(db_path, table):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_valid_transaction_is_stored_everywhere(db_path, make_transaction):
    status = ingestion.process_transaction(
        make_transaction(), db_path=db_path, is_fraud=1
    )

    assert status == "ACCEPTED"
    for table in [
        "transactions", "transaction_features",
        "fraud_predictions", "transaction_labels",
    ]:
        assert count(db_path, table) == 1
    assert count(db_path, "quarantine_transactions") == 0


def test_invalid_transaction_is_quarantined(db_path, make_transaction):
    status = ingestion.process_transaction(
        make_transaction(amount=-100), db_path=db_path
    )

    assert status == "REJECTED"
    assert count(db_path, "transactions") == 0
    with sqlite3.connect(db_path) as conn:
        reason = conn.execute(
            "SELECT rejection_reason FROM quarantine_transactions"
        ).fetchone()[0]
    assert reason == "amount must be a finite positive number"


def test_missing_field_is_quarantined_not_crashing(db_path, make_transaction):
    transaction = make_transaction()
    del transaction["customer_id"]

    assert ingestion.process_transaction(transaction, db_path=db_path) == "REJECTED"
    assert count(db_path, "quarantine_transactions") == 1


def test_duplicate_id_is_quarantined(db_path, make_transaction):
    first = make_transaction()
    duplicate = make_transaction(transaction_id=first["transaction_id"])

    assert ingestion.process_transaction(first, db_path=db_path) == "ACCEPTED"
    assert ingestion.process_transaction(duplicate, db_path=db_path) == "DUPLICATE"
    assert count(db_path, "transactions") == 1
    assert count(db_path, "quarantine_transactions") == 1


def test_timestamps_are_stored_in_utc(db_path, make_transaction):
    ingestion.process_transaction(
        make_transaction(timestamp="2026-11-24T17:30:00+05:30"),
        db_path=db_path,
    )
    with sqlite3.connect(db_path) as conn:
        stored = conn.execute("SELECT timestamp FROM transactions").fetchone()[0]
    assert stored == "2026-11-24T12:00:00.000000+00:00"


def test_database_uses_wal(db_path):
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_schema_mismatch_raises_instead_of_dropping(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE transaction_features (transaction_id TEXT PRIMARY KEY)"
        )
        conn.execute("INSERT INTO transaction_features VALUES ('keep-me')")
    conn.close()

    with pytest.raises(SchemaMismatchError):
        create_database(path)

    # The existing data must still be there.
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT transaction_id FROM transaction_features"
        ).fetchall() == [("keep-me",)]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_real_model_scores_a_transaction(monkeypatch, db_path, make_transaction):
    from src.fraud_predictor import predict_fraud

    monkeypatch.setattr(ingestion, "predict_fraud", predict_fraud)
    assert ingestion.process_transaction(
        make_transaction(), db_path=db_path
    ) == "ACCEPTED"
    with sqlite3.connect(db_path) as conn:
        score, version = conn.execute(
            "SELECT fraud_score, model_version FROM fraud_predictions"
        ).fetchone()
    assert 0.0 <= score <= 1.0
    assert version == "v2.0"
