from datetime import datetime, timedelta

from src.database import to_utc_iso

# Timestamps are stored as fixed-format UTC ISO strings (see
# database.to_utc_iso), so they compare correctly as plain text and the
# (customer_id, timestamp) index can be used.


def calculate_features(conn, transaction):
    customer_id = transaction["customer_id"]
    current_time = datetime.fromisoformat(to_utc_iso(transaction["timestamp"]))

    current_iso = to_utc_iso(current_time)
    five_minutes_ago = to_utc_iso(current_time - timedelta(minutes=5))
    one_hour_ago = to_utc_iso(current_time - timedelta(hours=1))

    # Customer transactions in the previous hour.
    recent = conn.execute("""
        SELECT amount, timestamp
        FROM transactions
        WHERE customer_id = ?
          AND timestamp >= ?
          AND timestamp < ?
    """, (customer_id, one_hour_ago, current_iso)).fetchall()

    # 1. Transaction count in previous 5 minutes
    txn_count_5m = sum(1 for _, ts in recent if ts >= five_minutes_ago)

    # 2. Total spending in previous hour
    amount_sum_1h = sum(amount for amount, _ in recent)

    # 3. Amount compared with the customer's historical average
    avg_prior_amount = conn.execute("""
        SELECT AVG(amount)
        FROM transactions
        WHERE customer_id = ?
          AND timestamp < ?
    """, (customer_id, current_iso)).fetchone()[0]

    amount_ratio = (
        transaction["amount"] / avg_prior_amount
        if avg_prior_amount is not None and avg_prior_amount > 0
        else None
    )

    # 4. Did the city change since the customer's last transaction?
    last_record = conn.execute("""
        SELECT city
        FROM transactions
        WHERE customer_id = ?
          AND timestamp < ?
        ORDER BY timestamp DESC, transaction_id DESC
        LIMIT 1
    """, (customer_id, current_iso)).fetchone()

    city_changed = (
        int(last_record[0] != transaction["city"])
        if last_record is not None
        else 0
    )

    return {
        "transaction_id": transaction["transaction_id"],
        "customer_id": customer_id,
        "amount": transaction["amount"],
        "txn_count_5m": txn_count_5m,
        "amount_sum_1h": round(amount_sum_1h, 2),
        "avg_prior_amount": avg_prior_amount,
        "amount_ratio": amount_ratio,
        "city_changed": city_changed,
        "feature_timestamp": current_iso,
    }
