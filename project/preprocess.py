import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/preprocess.log', level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

try:
    logging.info("Preprocessing started.")

    df = pd.read_csv('data/Online Retail.csv')
    logging.info(f"Raw data loaded with shape {df.shape}")

    # Calculate TotalPrice before dropping Quantity and UnitPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Drop columns that won't be used in segmentation
    df.drop(columns=['Quantity', 'UnitPrice', 'Country', 'StockCode', 'Description'], inplace=True)
    logging.info("Dropped unnecessary columns except CustomerID.")

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    logging.info(f"Duplicates dropped, new shape: {df.shape}")

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    logging.info("Converted InvoiceDate to datetime and set reference date.")

    # Drop rows with missing CustomerID before grouping
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)  # Ensure int type for merging
    
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
    rfm_scaled = pd.DataFrame(rfm_scaled_array, columns=['Recency', 'Frequency', 'Monetary'])
    rfm_scaled['CustomerID'] = rfm['CustomerID'].values
    rfm_scaled = rfm_scaled[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
    logging.info("RFM table scaled.")

    # Merge scaled RFM back to original data on CustomerID
    df = df.merge(rfm_scaled, on='CustomerID', how='left')
    logging.info("Scaled RFM merged with original data.")

    # Drop columns not needed for further modeling if any
    df.drop(columns=['InvoiceDate', 'InvoiceNo', 'TotalPrice'], inplace=True)
    logging.info("Dropped InvoiceDate, InvoiceNo, and TotalPrice from original data.")

    os.makedirs("data", exist_ok=True)
    df.to_csv('data/processed_data.csv', index=False)
    logging.info("Processed data saved to data/processed_data.csv")

except Exception as e:
    logging.error(f"Error occurred: {e}")
    raise e
