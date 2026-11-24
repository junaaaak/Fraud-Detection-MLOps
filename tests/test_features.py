from datetime import datetime, timedelta, timezone

import pytest

from src.database import insert_transaction, to_utc_iso
from src.feature_engineering import calculate_features

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


def at(minutes):
    return to_utc_iso(NOW + timedelta(minutes=minutes))


@pytest.fixture
def history(conn, make_transaction):
    """Prior transactions for CUST0001 at various offsets from NOW."""

    for minutes, amount, city in [
        (-120, 1000, "Bengaluru"),  # outside the 1h window
        (-30, 500, "Bengaluru"),    # inside 1h, outside 5m
        (-4, 1000, "Bengaluru"),    # inside 5m
        (-1, 1500, "Bengaluru"),    # inside 5m
    ]:
        insert_transaction(conn, make_transaction(
            amount=amount, city=city, timestamp=at(minutes),
        ))
    # Another customer's activity must not leak in.
    insert_transaction(conn, make_transaction(
        customer_id="CUST0002", amount=99999, timestamp=at(-1),
    ))
    conn.commit()


def test_features_from_history(conn, history, make_transaction):
    features = calculate_features(
        conn, make_transaction(amount=40000, city="Mumbai", timestamp=at(0))
    )

    assert features["txn_count_5m"] == 2
    assert features["amount_sum_1h"] == 3000
    assert features["avg_prior_amount"] == 1000
    assert features["amount_ratio"] == 40
    assert features["city_changed"] == 1


def test_same_city_is_not_a_change(conn, history, make_transaction):
    features = calculate_features(
        conn, make_transaction(city="Bengaluru", timestamp=at(0))
    )
    assert features["city_changed"] == 0


def test_future_transactions_are_ignored(conn, history, make_transaction):
    # Scoring a transaction "in the past" must only see earlier history.
    features = calculate_features(conn, make_transaction(timestamp=at(-10)))
    assert features["txn_count_5m"] == 0
    assert features["amount_sum_1h"] == 500


def test_first_transaction_has_no_history(conn, make_transaction):
    features = calculate_features(conn, make_transaction(timestamp=at(0)))

    assert features["txn_count_5m"] == 0
    assert features["amount_sum_1h"] == 0
    assert features["avg_prior_amount"] is None
    assert features["amount_ratio"] is None
    assert features["city_changed"] == 0


def test_non_utc_timestamp_is_handled(conn, history, make_transaction):
    # 17:30 IST == 12:00 UTC, so results match test_features_from_history.
    features = calculate_features(conn, make_transaction(
        amount=40000, city="Mumbai", timestamp="2026-11-24T17:30:00+05:30",
    ))
    assert features["txn_count_5m"] == 2
    assert features["amount_sum_1h"] == 3000
