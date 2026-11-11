import pandas as pd
from database import db_path
import sqlite3

with sqlite3.connect(db_path) as conn:
    df = pd.read_sql_query("SELECT * FROM transactions", conn)

print(df.head(10)) # Display the first 10 rows of the transactions table
