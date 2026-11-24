"""
Synthetic transaction stream with labelled fraud.

Each customer has a home city, a usual device and a typical spend.
Normal transactions stay close to that profile. Fraud episodes break it
in one or more ways the pipeline's features can detect:

- amount spike   -> high amount / amount_ratio
- new city       -> city_changed
- rapid burst    -> txn_count_5m / amount_sum_1h

The same simulator feeds both the live pipeline and the training data,
so the model is trained on the distribution it will see in production.
"""

import math
import random
import uuid
from datetime import datetime, timezone

from src.database import to_utc_iso

MERCHANTS = ["Amazon", "Flipkart", "Swiggy", "Zepto", "Myntra", "BigBasket"]
CITIES = ["Bengaluru", "Hyderabad", "Chennai", "Mumbai", "Delhi", "Pune", "Kolkata"]
DEVICES = ["Phone", "Laptop", "Tablet"]

# Average seconds between transactions in the live stream
# (ingestion.py sleeps this long between events).
EVENT_INTERVAL_SECONDS = 2.0


class TransactionSimulator:

    def __init__(self, n_customers=200, fraud_rate=0.02, seed=None):
        self.rng = random.Random(seed)
        # Seeded runs derive IDs from the RNG so they are reproducible;
        # live runs use uuid4. Either way IDs are full 128-bit UUIDs.
        self._seeded = seed is not None
        self.fraud_rate = fraud_rate
        self.customers = [
            self._make_customer(i) for i in range(1, n_customers + 1)
        ]
        # Remaining transactions of an in-progress fraud burst.
        self._pending = []

    def _make_customer(self, number):
        typical = self.rng.lognormvariate(math.log(2000), 0.7)
        return {
            "customer_id": f"CUST{number:04d}",
            "home_city": self.rng.choice(CITIES),
            "device": self.rng.choice(DEVICES),
            "typical_amount": min(max(typical, 200), 20000),
        }

    def _build(self, customer, now, amount, city, device):
        return {
            "transaction_id": (
                uuid.UUID(int=self.rng.getrandbits(128)).hex
                if self._seeded
                else uuid.uuid4().hex
            ),
            "customer_id": customer["customer_id"],
            "amount": round(amount, 2),
            "merchant": self.rng.choice(MERCHANTS),
            "city": city,
            "device": device,
            "timestamp": to_utc_iso(now),
        }

    def _normal(self, customer, now):
        amount = customer["typical_amount"] * self.rng.lognormvariate(0, 0.5)
        city = (
            customer["home_city"]
            if self.rng.random() < 0.95
            else self.rng.choice(CITIES)
        )
        device = (
            customer["device"]
            if self.rng.random() < 0.9
            else self.rng.choice(DEVICES)
        )
        return self._build(customer, now, amount, city, device)

    def _start_fraud(self, customer, now):
        spike = self.rng.random() < 0.7
        new_city = self.rng.random() < 0.6
        burst = self.rng.random() < 0.5
        if not (spike or new_city or burst):
            spike = True

        city = (
            self.rng.choice([c for c in CITIES if c != customer["home_city"]])
            if new_city
            else customer["home_city"]
        )
        device = self.rng.choice(DEVICES)

        def amount():
            multiplier = self.rng.uniform(4, 15) if spike else self.rng.uniform(0.8, 2)
            return customer["typical_amount"] * multiplier

        if burst:
            extra = self.rng.randint(2, 5)
            self._pending = [
                (customer, amount, city, device) for _ in range(extra)
            ]

        return self._build(customer, now, amount(), city, device)

    def next(self, now=None):
        """Return (transaction, is_fraud) for the next event."""

        now = now or datetime.now(timezone.utc)

        if self._pending:
            customer, amount, city, device = self._pending.pop(0)
            return self._build(customer, now, amount(), city, device), 1

        customer = self.rng.choice(self.customers)
        if self.rng.random() < self.fraud_rate:
            return self._start_fraud(customer, now), 1
        return self._normal(customer, now), 0


_default_simulator = None


def generate_transaction():
    """Single unlabelled transaction from a shared simulator."""

    global _default_simulator
    if _default_simulator is None:
        _default_simulator = TransactionSimulator()
    transaction, _ = _default_simulator.next()
    return transaction


if __name__ == "__main__":
    simulator = TransactionSimulator()
    for _ in range(5):
        transaction, is_fraud = simulator.next()
        print("FRAUD " if is_fraud else "NORMAL", transaction)
