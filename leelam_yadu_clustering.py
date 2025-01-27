import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def perform_clustering():
    # Load datasets
    customers = pd.read_csv('./data/Customers.csv')
    transactions = pd.read_csv('./data/Transactions.csv')

    # Convert date columns to datetime objects
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')
    customers['SignupDate'] = pd.to_datetime(customers['SignupDate'], errors='coerce')

    # Extract features from date columns
    transactions['TransactionYear'] = transactions['TransactionDate'].dt.year
    transactions['TransactionMonth'] = transactions['TransactionDate'].dt.month
    transactions['TransactionDay'] = transactions['TransactionDate'].dt.day

    customers['SignupYear'] = customers['SignupDate'].dt.year
    customers['SignupMonth'] = customers['SignupDate'].dt.month
    customers['SignupDay'] = customers['SignupDate'].dt.day

    # Drop the original date columns
    transactions = transactions.drop(columns=['TransactionDate'])
    customers = customers.drop(columns=['SignupDate'])

    # Aggregate customer transaction data
    customer_transactions = transactions.groupby('CustomerID').agg({
        'TotalValue': 'sum',
        'Quantity': 'sum',
        'TransactionYear': 'mean',
        'TransactionMonth': 'mean',
        'TransactionDay': 'mean'
    }).reset_index()

    # Merge customer and transaction data
    customer_profile = customers.merge(customer_transactions, on='CustomerID', how='left').fillna(0)

    # Encode categorical variables
    customer_profile_encoded = pd.get_dummies(customer_profile, columns=['Region'])

    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_profile['Cluster'] = kmeans.fit_predict(customer_profile_encoded.iloc[:, 2:])

    # Evaluate clustering with Davies-Bouldin Index
    db_index = davies_bouldin_score(customer_profile_encoded.iloc[:, 2:], customer_profile['Cluster'])
    print("Davies-Bouldin Index:", db_index)

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(customer_profile_encoded.iloc[:, 2:])
    customer_profile['PCA1'] = reduced_data[:, 0]
    customer_profile['PCA2'] = reduced_data[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=customer_profile, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
    plt.title('Customer Segments')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('./output/customer_segments.png')
    print("Clustering completed. Results saved in 'output' folder.")


if __name__ == '__main__':
    perform_clustering()
