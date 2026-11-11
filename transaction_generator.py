import random
import uuid
from datetime import datetime, time, timezone
import time

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

if __name__ == "__main__":
    transaction = generate_transaction()
    print(transaction)
    time.sleep(2)  # Sleep for 2 seconds before generating the next transaction