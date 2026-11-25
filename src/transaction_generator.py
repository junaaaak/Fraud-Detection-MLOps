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

Customer profiles come from a fixed seed, so a customer keeps the same
home city / device / spend across restarts and their stored history
stays meaningful. `history()` produces that back-history, and the live
stream can also emit malformed or duplicate records (`bad_data_rate`)
to exercise validation and quarantine.
"""

import math
import random
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone

from src.database import to_utc_iso

MERCHANTS = ["Amazon", "Flipkart", "Swiggy", "Zepto", "Myntra", "BigBasket"]
CITIES = ["Bengaluru", "Hyderabad", "Chennai", "Mumbai", "Delhi", "Pune", "Kolkata"]
DEVICES = ["Phone", "Laptop", "Tablet"]

# Average seconds between transactions in the live stream
# (ingestion.py sleeps this long between events).
EVENT_INTERVAL_SECONDS = 2.0

# Seed for customer profiles (independent of the event stream seed).
PROFILE_SEED = 2026

# Back-history: normal transactions per customer, placed this many days
# before "now" so it never appears in the dashboard's recent windows.
HISTORY_PER_CUSTOMER = 20
HISTORY_DAYS = (8, 30)

# Ways the live stream corrupts a record (each fails validation, except
# "duplicate", which is caught by the transaction_id primary key).
CORRUPTIONS = [
    "missing_field",
    "blank_field",
    "negative_amount",
    "zero_amount",
    "text_amount",
    "nan_amount",
    "bad_timestamp",
    "naive_timestamp",
    "duplicate",
]


class TransactionSimulator:

    def __init__(
        self,
        n_customers=200,
        fraud_rate=0.02,
        bad_data_rate=0.0,
        seed=None,
        profile_seed=PROFILE_SEED,
    ):
        self.rng = random.Random(seed)
        # Seeded runs derive IDs from the RNG so they are reproducible;
        # live runs use uuid4. Either way IDs are full 128-bit UUIDs.
        self._seeded = seed is not None
        self.fraud_rate = fraud_rate
        self.bad_data_rate = bad_data_rate
        profile_rng = random.Random(profile_seed)
        self.customers = [
            self._make_customer(profile_rng, i)
            for i in range(1, n_customers + 1)
        ]
        # Remaining transactions of an in-progress fraud burst.
        self._pending = []
        # Recently emitted valid transactions, for duplicate re-sends.
        self._recent = deque(maxlen=50)

    @staticmethod
    def _make_customer(rng, number):
        typical = rng.lognormvariate(math.log(2000), 0.7)
        return {
            "customer_id": f"CUST{number:04d}",
            "home_city": rng.choice(CITIES),
            "device": rng.choice(DEVICES),
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

    def _corrupt(self, customer, now):
        """A record that the pipeline should quarantine."""

        kind = self.rng.choice(CORRUPTIONS)
        if kind == "duplicate" and self._recent:
            return dict(self.rng.choice(self._recent))
        if kind == "duplicate":
            kind = "missing_field"

        transaction = self._normal(customer, now)
        if kind == "missing_field":
            del transaction[self.rng.choice(
                ["customer_id", "amount", "merchant", "city", "device", "timestamp"]
            )]
        elif kind == "blank_field":
            transaction[self.rng.choice(["customer_id", "merchant", "city"])] = "  "
        elif kind == "negative_amount":
            transaction["amount"] = -transaction["amount"]
        elif kind == "zero_amount":
            transaction["amount"] = 0
        elif kind == "text_amount":
            transaction["amount"] = f"Rs.{transaction['amount']}"
        elif kind == "nan_amount":
            transaction["amount"] = float("nan")
        elif kind == "bad_timestamp":
            transaction["timestamp"] = now.strftime("%d/%m/%Y %H:%M")
        elif kind == "naive_timestamp":
            transaction["timestamp"] = now.replace(tzinfo=None).isoformat()
        return transaction

    def history(self, end=None, per_customer=HISTORY_PER_CUSTOMER):
        """Normal back-history for every customer, oldest first.

        Placed HISTORY_DAYS before `end`, so features like amount_ratio
        and city_changed have something to compare against from the
        first live transaction.
        """

        end = end or datetime.now(timezone.utc)
        low, high = HISTORY_DAYS
        rows = []
        for customer in self.customers:
            for _ in range(per_customer):
                when = end - timedelta(days=self.rng.uniform(low, high))
                rows.append(self._normal(customer, when))
        rows.sort(key=lambda t: t["timestamp"])
        return rows

    def next(self, now=None):
        """Return (transaction, is_fraud) for the next event.

        is_fraud is None for corrupted / duplicate records, which are
        expected to be quarantined rather than scored.
        """

        now = now or datetime.now(timezone.utc)

        if self._pending:
            customer, amount, city, device = self._pending.pop(0)
            transaction, label = self._build(customer, now, amount(), city, device), 1
        else:
            customer = self.rng.choice(self.customers)
            if self.bad_data_rate and self.rng.random() < self.bad_data_rate:
                return self._corrupt(customer, now), None
            if self.rng.random() < self.fraud_rate:
                transaction, label = self._start_fraud(customer, now), 1
            else:
                transaction, label = self._normal(customer, now), 0

        self._recent.append(transaction)
        return transaction, label


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
