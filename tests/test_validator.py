import math

import pytest

from src.validator import REQUIRED_FIELDS, validate_transaction


def test_valid_transaction_passes(make_transaction):
    assert validate_transaction(make_transaction()) == (True, [])


def test_non_dict_is_rejected():
    is_valid, errors = validate_transaction("not a dict")
    assert not is_valid
    assert errors == ["Transaction must be a dictionary"]


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_field_is_checked(make_transaction, field):
    transaction = make_transaction()
    del transaction[field]

    is_valid, errors = validate_transaction(transaction)

    assert not is_valid
    assert errors == [f"{field} is required"]


@pytest.mark.parametrize("field", ["customer_id", "merchant", "city", "device"])
def test_whitespace_strings_are_rejected(make_transaction, field):
    is_valid, errors = validate_transaction(make_transaction(**{field: "   "}))
    assert not is_valid
    assert errors == [f"{field} cannot be empty or whitespace"]


@pytest.mark.parametrize("field", ["transaction_id", "customer_id", "city"])
def test_non_string_fields_are_rejected(make_transaction, field):
    is_valid, errors = validate_transaction(make_transaction(**{field: 123}))
    assert not is_valid
    assert errors == [f"{field} must be a string"]


@pytest.mark.parametrize(
    "amount", [-100.0, 0, math.nan, math.inf, True, "500"]
)
def test_invalid_amounts_are_rejected(make_transaction, amount):
    is_valid, errors = validate_transaction(make_transaction(amount=amount))
    assert not is_valid
    assert errors == ["amount must be a finite positive number"]


@pytest.mark.parametrize("amount", [0.01, 1, 49999.99])
def test_valid_amounts_pass(make_transaction, amount):
    assert validate_transaction(make_transaction(amount=amount))[0]


@pytest.mark.parametrize("timestamp, error", [
    ("2026-11-24T10:00:00", "timestamp must include a timezone"),
    ("yesterday", "timestamp has an invalid format"),
    (1732440000, "timestamp must be ISO datetime text"),
])
def test_invalid_timestamps_are_rejected(make_transaction, timestamp, error):
    is_valid, errors = validate_transaction(make_transaction(timestamp=timestamp))
    assert not is_valid
    assert errors == [error]


def test_non_utc_offset_is_accepted(make_transaction):
    transaction = make_transaction(timestamp="2026-11-24T15:30:00+05:30")
    assert validate_transaction(transaction)[0]


def test_multiple_problems_are_all_reported(make_transaction):
    transaction = make_transaction(amount=-5)
    del transaction["customer_id"]

    is_valid, errors = validate_transaction(transaction)

    assert not is_valid
    assert set(errors) == {
        "customer_id is required",
        "amount must be a finite positive number",
    }
