import pandas as pd
import logging
import os
import joblib
import numpy as np
import yaml
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Setup logging and directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
logging.basicConfig(filename='logs/train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

try:
    logging.info("Training started.")

    # Load params.yaml
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    kmeans_params = params['kmeans']
    dbscan_params = params['dbscan']
    hier_params = params['hierarchical']

    # Load processed data
    df = pd.read_csv("data/processed_data.csv")
    logging.info(f"Processed data loaded with shape {df.shape}")

    customer_ids = df['CustomerID']
    X = df.drop(columns=['CustomerID'])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=kmeans_params['n_clusters'], random_state=kmeans_params['random_state'])
    kmeans_labels = kmeans.fit_predict(X)
    df['KMeans_Cluster'] = kmeans_labels
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    logging.info(f"KMeans clustering completed with silhouette score: {kmeans_silhouette:.4f}")

    # PCA for DBSCAN and Hierarchical Clustering
    pca = PCA(n_components=2, random_state=kmeans_params['random_state'])
    X_reduced = pca.fit_transform(X)

    # DBSCAN Clustering
    dbscan_sample_size = dbscan_params['sample_size']
    if len(X_reduced) > dbscan_sample_size:
        dbscan_indices = np.random.choice(len(X_reduced), dbscan_sample_size, replace=False)
    else:
        dbscan_indices = np.arange(len(X_reduced))
    X_dbscan_sample = X_reduced[dbscan_indices]

    dbscan = DBSCAN(eps=dbscan_params['eps'], min_samples=dbscan_params['min_samples'])
    dbscan_labels_sample = dbscan.fit_predict(X_dbscan_sample)

    dbscan_labels_full = -1 * np.ones(len(X_reduced), dtype=int)
    dbscan_labels_full[dbscan_indices] = dbscan_labels_sample
    df['DBSCAN_Cluster'] = dbscan_labels_full

    mask = dbscan_labels_sample != -1
    if mask.sum() > 0:
        dbscan_silhouette = silhouette_score(X_dbscan_sample[mask], dbscan_labels_sample[mask])
        logging.info(f"DBSCAN clustering (sampled) silhouette score: {dbscan_silhouette:.4f}")
    else:
        logging.info("DBSCAN found no clusters (all noise).")

    # Hierarchical Clustering
    hier_sample_size = hier_params['sample_size']
    if len(X_reduced) > hier_sample_size:
        hier_indices = np.random.choice(len(X_reduced), hier_sample_size, replace=False)
    else:
        hier_indices = np.arange(len(X_reduced))
    X_hier_sample = X_reduced[hier_indices]

    Z = linkage(X_hier_sample, method='ward')
    hier_labels_sample = fcluster(Z, t=hier_params['n_clusters'], criterion='maxclust')

    hier_labels_full = -1 * np.ones(len(X_reduced), dtype=int)
    hier_labels_full[hier_indices] = hier_labels_sample
    df['Hierarchical_Cluster'] = hier_labels_full

    hier_silhouette = silhouette_score(X_hier_sample, hier_labels_sample)
    logging.info(f"Hierarchical clustering (sampled) silhouette score: {hier_silhouette:.4f}")

    # Save clustered data and models
    df['CustomerID'] = customer_ids
    df.to_csv('data/clustered_data.csv', index=False)
    logging.info("Clustered data saved to 'data/clustered_data.csv'.")

    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(dbscan, 'models/dbscan_model.pkl')
    joblib.dump(Z, 'models/hierarchical_model.pkl')
    logging.info("Models saved successfully.")

except Exception as e:
    logging.error(f"Error occurred: {e}")
    raise e
