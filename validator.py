from datetime import datetime
import math


REQUIRED_FIELDS = (
    "transaction_id",
    "customer_id",
    "amount",
    "merchant",
    "city",
    "device",
    "timestamp",
)

STRING_FIELDS = ("transaction_id", "customer_id", "merchant", "city", "device")


def validate_transaction(transaction):
    """Return whether a transaction is valid and a list of validation errors."""
    if not isinstance(transaction, dict):
        return False, ["Transaction must be a dictionary"]

    errors = []

    for field in REQUIRED_FIELDS:
        value = transaction.get(field)
        if value is None or value == "":
            errors.append(f"{field} is required")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"{field} cannot be empty or whitespace")

    for field in STRING_FIELDS:
        value = transaction.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"{field} must be a string")

    amount = transaction.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        errors.append("amount must be a finite non-negative number")
    else:
        try:
            finite_amount = math.isfinite(amount)
        except OverflowError:
            finite_amount = False
        if not finite_amount or amount < 0:
            errors.append("amount must be a finite non-negative number")

    timestamp = transaction.get("timestamp")
    if not isinstance(timestamp, str):
        errors.append("timestamp must be an ISO datetime string")
    else:
        try:
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                errors.append("timestamp must include a timezone")
        except ValueError:
            errors.append("timestamp has an invalid format")

    return not errors, errors
