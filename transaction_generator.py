import random
import uuid
from datetime import datetime, timezone

customer=["CUST101","CUST102","CUST103","CUST104","CUST105"]
merchant=["Flipkart","Amazon","Myntra","Bigbasket","Swiggy"]
cities=["Bangalore","Chennai","Mumbai","Delhi","Kolkata"]
devices=["Mobile","Laptop","Tablet"]

def generate_transaction():
    return {
        "transaction_id": str(uuid.uuid4())[:8],
        "customer_id": random.choice(customer),
        "merchant": random.choice(merchant),
        "amount": round(random.uniform(500, 5000), 2),
        "city": random.choice(cities),
        "device": random.choice(devices),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

for i in range(10):
    transaction = generate_transaction()
    print(transaction)
