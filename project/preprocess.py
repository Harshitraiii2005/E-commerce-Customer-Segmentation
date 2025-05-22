import pandas as pd
import numpy as np
import logging
import os
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

sample_size = params['preprocess'].get('sample_size', None)
pca_components = params['preprocess'].get('pca_components', None)

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/preprocess.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

try:
    logging.info("Preprocessing started.")

    df = pd.read_csv('data/Online Retail.csv')
    logging.info(f"Raw data loaded with shape {df.shape}")

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logging.info(f"Sampled {sample_size} rows from the dataset.")

    # Calculate TotalPrice before dropping Quantity and UnitPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Drop columns not used in modeling
    df.drop(columns=['Quantity', 'UnitPrice', 'Country', 'StockCode', 'Description'], inplace=True)
    logging.info("Dropped unnecessary columns.")

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    logging.info(f"Duplicates dropped, new shape: {df.shape}")

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Drop rows with missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Create RFM table
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    logging.info("RFM table created.")

    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled_array = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Apply PCA if specified
    if pca_components and pca_components < 3:
        pca = PCA(n_components=pca_components)
        rfm_scaled_array = pca.fit_transform(rfm_scaled_array)
        cols = [f'PC{i+1}' for i in range(pca_components)]
    else:
        cols = ['Recency', 'Frequency', 'Monetary']

    rfm_scaled = pd.DataFrame(rfm_scaled_array, columns=cols)
    rfm_scaled['CustomerID'] = rfm['CustomerID'].values

    # Merge back with original
    df = df.merge(rfm_scaled, on='CustomerID', how='left')

    # Drop unused columns
    df.drop(columns=['InvoiceDate', 'InvoiceNo', 'TotalPrice'], inplace=True)

    # Save processed data
    os.makedirs("data", exist_ok=True)
    df.to_csv('data/processed_data.csv', index=False)
    logging.info("Processed data saved to data/processed_data.csv")

except Exception as e:
    logging.error(f"Error occurred: {e}")
    raise e
