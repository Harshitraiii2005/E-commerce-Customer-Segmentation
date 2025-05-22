import pandas as pd
import logging
import joblib
import numpy as np
import json
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

logging.basicConfig(filename='logs/evaluate.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    try:
        # Load clustered data
        df = pd.read_csv("data/clustered_data.csv")
        logging.info(f"Loaded clustered data with shape {df.shape}")
        
        feature_cols = ['PC1', 'PC2']
        for col in feature_cols:
            if col not in df.columns:
                raise KeyError(f"Expected feature column '{col}' not found in data.")
        
        X = df[feature_cols].values
        
        results = {}
        
        # Helper function to compute all metrics if possible
        def compute_metrics(X_sub, labels, method_name):
            metrics = {}
            n_clusters = len(np.unique(labels[labels != -1]))
            if n_clusters < 2:
                logging.info(f"{method_name}: Not enough clusters to compute metrics.")
                return None
            
            metrics['silhouette_score'] = silhouette_score(X_sub, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_sub, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_sub, labels)
            logging.info(f"{method_name} scores: {metrics}")
            return metrics
        
        # KMeans evaluation (no noise label expected)
        kmeans_labels = df['KMeans_Cluster'].values
        results['KMeans'] = compute_metrics(X, kmeans_labels, 'KMeans')
        
        # DBSCAN evaluation (exclude noise = -1)
        dbscan_labels = df['DBSCAN_Cluster'].values
        mask_dbscan = dbscan_labels != -1
        if np.sum(mask_dbscan) > 1:
            results['DBSCAN'] = compute_metrics(X[mask_dbscan], dbscan_labels[mask_dbscan], 'DBSCAN')
        else:
            logging.info("DBSCAN clustering has insufficient clusters or only noise.")
            results['DBSCAN'] = None
        
        # Hierarchical evaluation (exclude noise = -1)
        hier_labels = df['Hierarchical_Cluster'].values
        mask_hier = hier_labels != -1
        if np.sum(mask_hier) > 1:
            results['Hierarchical'] = compute_metrics(X[mask_hier], hier_labels[mask_hier], 'Hierarchical')
        else:
            logging.info("Hierarchical clustering has insufficient clusters or only noise.")
            results['Hierarchical'] = None
        
        # Make sure metrics directory exists
        os.makedirs('metrics', exist_ok=True)
        
        # Save metrics to JSON
        with open('metrics/evaluation_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info("Evaluation metrics saved to 'metrics/evaluation_metrics.json'.")
        print("Evaluation completed successfully. Metrics saved.")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")
        raise e

if __name__ == "__main__":
    main()
