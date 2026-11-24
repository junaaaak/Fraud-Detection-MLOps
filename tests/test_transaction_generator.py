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
