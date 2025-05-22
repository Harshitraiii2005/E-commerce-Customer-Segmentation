import pandas as pd
import logging
import os
import joblib  # corrected import
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
logging.basicConfig(filename='logs/train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

try:
    logging.info("Training started.")

    df = pd.read_csv("data/processed_data.csv")
    logging.info(f"Processed data loaded with shape {df.shape}")

    # Save CustomerID separately if needed later
    customer_ids = df['CustomerID']
    X = df.drop(columns=['CustomerID'])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    df['KMeans_Cluster'] = kmeans_labels
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    logging.info(f"KMeans clustering completed with silhouette score: {kmeans_silhouette:.4f}")

    # PCA for dimensionality reduction before DBSCAN
    pca = PCA(n_components=5, random_state=42)
    X_reduced = pca.fit_transform(X)

    # DBSCAN on reduced data
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_reduced)
    df['DBSCAN_Cluster'] = dbscan_labels

    # Silhouette score on reduced data excluding noise
    mask = dbscan_labels != -1
    if mask.sum() > 0:
        dbscan_silhouette = silhouette_score(X_reduced[mask], dbscan_labels[mask])
        logging.info(f"DBSCAN clustering completed with silhouette score (excluding noise): {dbscan_silhouette:.4f}")
    else:
        logging.info("DBSCAN found no clusters (all points marked as noise)")

    # Hierarchical Clustering
    Z = linkage(X, method='ward')
    hier_labels = fcluster(Z, t=5, criterion='maxclust')
    df['Hierarchical_Cluster'] = hier_labels
    hier_silhouette = silhouette_score(X, hier_labels)
    logging.info(f"Hierarchical clustering completed with silhouette score: {hier_silhouette:.4f}")

    # Save clustered data with CustomerID back
    df['CustomerID'] = customer_ids
    df.to_csv('data/clustered_data.csv', index=False)
    logging.info("Clustered data saved to 'data/clustered_data.csv'.")

    # Save models
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(dbscan, 'models/dbscan_model.pkl')
    joblib.dump(Z, 'models/hierarchical_model.pkl')
    logging.info("Models saved successfully.")

except Exception as e:
    logging.error(f"Error occurred: {e}")
    raise e
