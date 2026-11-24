import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from src.database import create_database, to_utc_iso


@pytest.fixture
def db_path(tmp_path):
    """A fresh database file per test; never touches data/database.db."""

    path = tmp_path / "test.db"
    create_database(path)
    return path


@pytest.fixture
def conn(db_path):
    connection = sqlite3.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture
def make_transaction():
    def _make(**overrides):
        transaction = {
            "transaction_id": uuid.uuid4().hex,
            "customer_id": "CUST0001",
            "amount": 1500.0,
            "merchant": "Amazon",
            "city": "Bengaluru",
            "device": "Phone",
            "timestamp": to_utc_iso(datetime.now(timezone.utc)),
        }
        transaction.update(overrides)
        return transaction

    return _make
