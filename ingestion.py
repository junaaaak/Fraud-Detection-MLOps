import sqlite3
import time

from database import db_path, create_database
from transaction_generator import generate_transaction

create_database()
while True:
    transaction = generate_transaction()
    with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO transactions (transaction_id, customer_id, merchant, amount, city, device, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    transaction["transaction_id"],
                    transaction["customer_id"],
                    transaction["merchant"],
                    transaction["amount"],
                    transaction["city"],
                    transaction["device"],
                    transaction["timestamp"],
                ),
            )
    print(f"Saved {transaction['customer_id']}",
          f"| amount: {transaction['amount']}",
          f"| merchant: {transaction['merchant']}",)

    time.sleep(2)  # Sleep for 2 seconds before generating the next transaction