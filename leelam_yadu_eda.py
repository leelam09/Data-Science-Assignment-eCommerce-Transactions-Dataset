import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda():
    # Load datasets
    try:
        customers = pd.read_csv('./data/Customers.csv')
        products = pd.read_csv('./data/Products.csv')
        transactions = pd.read_csv('./data/Transactions.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check if the data files exist.")
        return

    # Convert dates to datetime
    if 'SignupDate' in customers.columns:
        customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
    else:
        print("Warning: 'SignupDate' column not found in customers data.")

    if 'TransactionDate' in transactions.columns:
        transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    else:
        print("Warning: 'TransactionDate' column not found in transactions data.")

    # Check for missing values
    print("\nMissing Values:")
    print("Customers:", customers.isnull().sum())
    print("Products:", products.isnull().sum())
    print("Transactions:", transactions.isnull().sum())

    # Analyze customers by region
    plt.figure(figsize=(10, 5))
    sns.countplot(data=customers, x='Region', hue='Region')  # Added hue to fix the FutureWarning
    plt.title('Customer Distribution by Region')
    plt.xlabel('Region')
    plt.ylabel('Count')
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(f'{output_dir}/customer_distribution_by_region.png')
    plt.close()

    # Top-selling products
    top_products = transactions.groupby('ProductID')['Quantity'].sum().reset_index()
    top_products = top_products.merge(products, on='ProductID').sort_values(by='Quantity', ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_products.head(10), x='ProductName', y='Quantity', hue='Category')  # Added hue for color based on Category
    plt.title('Top 10 Products by Quantity Sold')
    plt.xlabel('Product Name')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/top_10_products.png')
    plt.close()

    print("EDA completed. Plots saved in the 'output' folder.")
    
    # Handle missing values (example: drop rows with missing values)
    customers.dropna(inplace=True)
    products.dropna(inplace=True)
    transactions.dropna(inplace=True)


if __name__ == '__main__':
    perform_eda()
