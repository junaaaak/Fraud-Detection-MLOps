import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "data" / "database.db"

def create_database():
    """Create the SQLite database and the necessary tables."""

    db_path.parent.mkdir(parents=True, exist_ok=True)  # Create the data directory if it doesn't exist

    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT,
                merchant TEXT,
                amount REAL,
                city TEXT,
                device TEXT,
                timestamp TEXT
            )
        ''')
    print ("Database is ready at:", db_path)

if __name__ == "__main__":
    create_database()