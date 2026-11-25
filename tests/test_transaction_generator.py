from src.transaction_generator import TransactionSimulator, generate_transaction
from src.validator import validate_transaction


def test_simulated_transactions_are_valid():
    simulator = TransactionSimulator(seed=1)
    for _ in range(500):
        transaction, _ = simulator.next()
        assert validate_transaction(transaction) == (True, [])


def test_fraud_rate_is_realistic():
    simulator = TransactionSimulator(seed=1)
    labels = [simulator.next()[1] for _ in range(5000)]
    rate = sum(labels) / len(labels)
    assert 0.01 < rate < 0.15


def test_seeded_simulator_is_reproducible():
    a = TransactionSimulator(seed=7)
    b = TransactionSimulator(seed=7)
    for _ in range(50):
        ta, la = a.next()
        tb, lb = b.next()
        assert la == lb
        assert {k: v for k, v in ta.items() if k != "timestamp"} == \
               {k: v for k, v in tb.items() if k != "timestamp"}


def test_transaction_ids_are_full_uuids():
    assert len(generate_transaction()["transaction_id"]) == 32


def test_customer_profiles_are_stable_across_runs():
    a = TransactionSimulator(seed=1).customers
    b = TransactionSimulator(seed=99).customers
    c = TransactionSimulator().customers
    assert a == b == c


def test_history_gives_every_customer_past_transactions():
    from datetime import datetime, timezone

    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    simulator = TransactionSimulator(n_customers=10, seed=3)
    rows = simulator.history(end=end, per_customer=5)
    assert len(rows) == 50
    assert {t["customer_id"] for t in rows} == {
        c["customer_id"] for c in simulator.customers
    }
    assert all(t["timestamp"] < end.isoformat() for t in rows)
    assert all(validate_transaction(t)[0] for t in rows)


def test_bad_data_is_rejected_or_duplicate():
    simulator = TransactionSimulator(seed=5, bad_data_rate=0.3)
    seen, invalid, duplicates = set(), 0, 0
    for _ in range(2000):
        transaction, label = simulator.next()
        valid, _ = validate_transaction(transaction)
        if label is None:
            if valid:
                assert transaction["transaction_id"] in seen
                duplicates += 1
            else:
                invalid += 1
        else:
            assert valid
            seen.add(transaction["transaction_id"])
    assert invalid > 0 and duplicates > 0


def test_clean_stream_has_no_bad_data():
    simulator = TransactionSimulator(seed=5)
    assert all(simulator.next()[1] is not None for _ in range(2000))
