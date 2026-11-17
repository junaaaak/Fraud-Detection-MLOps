import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

n=1000  # Number of transactions to generate

# generate synthetic transaction data
amount = rng.uniform(100, 50000, n)

txn_count_5m = rng.poisson(5, n)
amount_sum_1h = rng.uniform(0, 120000, n)

amount_ratio = rng.lognormal(mean=0.5, sigma=1, size=n)

city_changed = rng.binomial(1, 0.2, n)

#simulate hidden relationships between features and target variable
risk = (
    -4.0
    +0.75 * (txn_count_5m >=3)
    +1.3 * (amount_ratio > 5)
    +1.1 * (amount_ratio > 15)
    +0.5 * city_changed
    +0.5 * (amount>25000)
    +0.5 * (amount_sum_1h>40000)
)

fraud_probability = 1 / (1 + np.exp(-risk)) #sigmoid function to convert risk score to probability

fraud_label = rng.binomial(1, fraud_probability)

df = pd.DataFrame({
    "amount": amount,
    "txn_count_5m": txn_count_5m,
    "amount_sum_1h": amount_sum_1h,
    "amount_ratio": amount_ratio,
    "city_changed": city_changed,
    "fraud_probability": fraud_probability,
    "fraud_label": fraud_label
})

Path("data").mkdir(parents=True, exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print("Synthetic transaction data generated and saved to 'data/transactions.csv'")
print("Total transactions:", len(df))
print("Fraudulent transactions:", df["fraud_label"].sum())
print("Sample of generated data:")
print(df.head())