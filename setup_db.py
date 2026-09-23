import sqlite3
import pandas as pd
import numpy as np

print("Setting up SQLite Database...")

# Connect to (or create) SQLite database
conn = sqlite3.connect("retail_data.db")

# 1. Create Dimension Table: Products
np.random.seed(42)
product_ids = [f"PROD_{i:03d}" for i in range(1, 51)]
categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Toys']

products_df = pd.DataFrame({
    'product_id': product_ids,
    'product_name': [f"Item {i}" for i in range(1, 51)],
    'category': np.random.choice(categories, size=50)
})

products_df.to_sql("products", conn, if_exists="replace", index=False)
print("Table 'products' created successfully!")

# 2. Create Fact Table: Transactions (1 year of daily transaction data)
dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
transactions_list = []

order_id = 1000
for d in dates:
    # Generate 10 to 30 random orders per day
    num_orders = np.random.randint(10, 30)
    for _ in range(num_orders):
        p_id = np.random.choice(product_ids)
        qty = np.random.randint(1, 5)
        price = round(np.random.uniform(10.0, 150.0), 2)
        cust_id = f"CUST_{np.random.randint(100, 300)}"
        
        transactions_list.append({
            'order_id': order_id,
            'order_date': d.strftime('%Y-%m-%d'),
            'product_id': p_id,
            'customer_id': cust_id,
            'quantity': qty,
            'unit_price': price
        })
        order_id += 1

transactions_df = pd.DataFrame(transactions_list)
transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)
print("Table 'transactions' created successfully!")

conn.close()
print("\nDatabase setup complete: 'retail_data.db' is ready!")
