from datetime import datetime
import math

REQUIRED_FIELDS = [
    "transaction_id",
    "customer_id",
    "amount",
    "merchant",
    "city",
    "device",
    "timestamp",
]

STRING_FIELDS = [
    "transaction_id",
    "customer_id",
    "merchant",
    "city",
    "device",
]


def validate_transaction(transaction):
    if not isinstance(transaction, dict):
        return False, ["Transaction must be a dictionary"]

    errors = []

    # 1. Required fields must be present and non-empty.
    missing = set()
    for field in REQUIRED_FIELDS:
        value = transaction.get(field)
        if value is None or value == "":
            errors.append(f"{field} is required")
            missing.add(field)
        elif isinstance(value, str) and not value.strip():
            errors.append(f"{field} cannot be empty or whitespace")
            missing.add(field)

    # 2. ID and descriptive fields must be strings.
    for field in STRING_FIELDS:
        value = transaction.get(field)
        if field not in missing and not isinstance(value, str):
            errors.append(f"{field} must be a string")

    # 3. Amount must be a finite, positive number.
    if "amount" not in missing:
        amount = transaction["amount"]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            errors.append("amount must be a finite positive number")

    # 4. Timestamp must be timezone-aware ISO 8601 text.
    if "timestamp" not in missing:
        timestamp = transaction["timestamp"]
        if not isinstance(timestamp, str):
            errors.append("timestamp must be ISO datetime text")
        else:
            try:
                parsed = datetime.fromisoformat(timestamp)
                if parsed.tzinfo is None:
                    errors.append("timestamp must include a timezone")
            except ValueError:
                errors.append("timestamp has an invalid format")

    return len(errors) == 0, errors
