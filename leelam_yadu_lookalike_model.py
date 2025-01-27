import sys
sys.tracebacklimit = 0
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def build_lookalike_model():
    try:
        # Load the dataset
        # Specify the correct delimiter if necessary
        customer_profile_encoded = pd.read_csv('data/customer_profile_encoded.csv', delimiter=',', on_bad_lines='skip')
        print("CSV file loaded successfully.")
        
        # Print the DataFrame and its data types
        print(customer_profile_encoded.head())  # Display the first few rows
        print(customer_profile_encoded.dtypes)  # Display the data types of the columns
        
        # Check if the DataFrame is empty
        if customer_profile_encoded.empty:
            print("The loaded DataFrame is empty. Please check the CSV file.")
            return
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        return

    # Check for the existence of 'SignupDate' column
    non_numeric_columns = customer_profile_encoded.select_dtypes(exclude=['number']).columns
    if 'SignupDate' in non_numeric_columns:
        # Handle date column (e.g., 'SignupDate')
        customer_profile_encoded['SignupDate'] = pd.to_datetime(customer_profile_encoded['SignupDate'], errors='coerce')
        customer_profile_encoded['SignupDays'] = (customer_profile_encoded['SignupDate'] - customer_profile_encoded['SignupDate'].min()).dt.days
        
        # Add 'SignupDays' to numeric columns
        numeric_columns = customer_profile_encoded.select_dtypes(include=['number']).columns.append(pd.Index(['SignupDays']))
        
        # Drop the original date column
        customer_profile_encoded.drop(columns=['SignupDate'], inplace=True)
    else:
        numeric_columns = customer_profile_encoded.select_dtypes(include=['number']).columns

    # Check if there are numeric columns before scaling
    if numeric_columns.empty:
        print("No numeric columns found for scaling. Please check the data.")
        return

    # Scale numeric columns
    scaler = StandardScaler()
    try:
        normalized_data = scaler.fit_transform(customer_profile_encoded[numeric_columns])
        print("Data scaled successfully.")
    except KeyError as e:
        print(f"Error with scaling: {e}")
        return

    # Continue with further modeling...
    print("Lookalike model preprocessing completed successfully.")


if __name__ == '__main__':
    build_lookalike_model()
