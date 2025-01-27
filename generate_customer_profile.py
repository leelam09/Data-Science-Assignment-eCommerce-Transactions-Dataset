import os
import pandas as pd

# Path to the customer profile encoded file
customer_profile_path = 'data/customer_profile_encoded.csv'

# Load the necessary input files
customers = pd.read_csv('data/Customers.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Check if the necessary columns exist in the data
if 'CustomerID' not in customers.columns or 'TransactionID' not in transactions.columns:
    raise ValueError("Missing required columns in the input files.")

# Create a copy of the customers DataFrame
customer_profile_encoded = customers.copy()

# Calculate TotalTransactions for each customer
total_transactions = transactions.groupby('CustomerID')['TransactionID'].count().reset_index(name='TotalTransactions')
customer_profile_encoded = customer_profile_encoded.merge(total_transactions, on='CustomerID', how='left')

# Handle missing values for 'TotalTransactions' if any
customer_profile_encoded['TotalTransactions'] = customer_profile_encoded['TotalTransactions'].fillna(0)

# Ensure the TotalTransactions column is numeric
customer_profile_encoded['TotalTransactions'] = pd.to_numeric(customer_profile_encoded['TotalTransactions'], errors='coerce')

# Save the generated file
os.makedirs('data', exist_ok=True)
customer_profile_encoded.to_csv(customer_profile_path, index=False)
print(f"Generated {customer_profile_path}.")
