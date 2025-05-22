import pandas as pd
import joblib 
import json
import os
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Ensure necessary folders exist
for folder in ['logs', 'models', 'metrics']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Logging setup
logging.basicConfig(
    filename='logs/evaluate.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def evaluate_model(X, labels, model_name):
    results = {}

    # Compute scores only if there are at least 2 unique clusters
    if len(set(labels)) < 2:
        logging.warning(f"{model_name}: Not enough clusters to evaluate metrics.")
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None
        }

    results['silhouette_score'] = silhouette_score(X, labels)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    results['davies_bouldin_score'] = davies_bouldin_score(X, labels)

    logging.info(f"{model_name} Evaluation Metrics:")
    logging.info(f"Silhouette Score: {results['silhouette_score']:.4f}")
    logging.info(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    logging.info(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    print(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")

    return results

def main():
    try:
        df = pd.read_csv("data/clustered_data.csv")
        logging.info(f"Clustered data loaded with shape {df.shape}")

        X = df[['Recency', 'Frequency', 'Monetary']].values

        # Load models (even if some are not directly reused)
        kmeans = joblib.load('models/kmeans_model.pkl')
        dbscan = joblib.load('models/dbscan_model.pkl')
        Z = joblib.load('models/hierarchical_model.pkl')
        logging.info("Models loaded successfully.")

        metrics = {}

        # Evaluate KMeans
        kmeans_labels = df.get('KMeans_Cluster', pd.Series([-1]*len(df))).values
        metrics['KMeans'] = evaluate_model(X, kmeans_labels, "KMeans")

        # Evaluate DBSCAN
        dbscan_labels = df.get('DBSCAN_Cluster', pd.Series([-1]*len(df))).values
        mask_dbscan = dbscan_labels != -1
        if mask_dbscan.sum() > 0:
            metrics['DBSCAN'] = evaluate_model(X[mask_dbscan], dbscan_labels[mask_dbscan], "DBSCAN")
        else:
            logging.warning("DBSCAN: No valid clusters found (all labeled as noise).")
            metrics['DBSCAN'] = {
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None
            }

        # Evaluate Hierarchical (sampled)
        hier_labels = df.get('Hierarchical_Cluster', pd.Series([-1]*len(df))).values
        mask_hier = hier_labels != -1
        if mask_hier.sum() > 0:
            metrics['Hierarchical'] = evaluate_model(X[mask_hier], hier_labels[mask_hier], "Hierarchical")
        else:
            logging.warning("Hierarchical: No valid clusters found in sampled labels.")
            metrics['Hierarchical'] = {
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None
            }

        # Save metrics to JSON
        with open('metrics/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Evaluation metrics saved to 'metrics/evaluation_metrics.json'.")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")
        raise e

if __name__ == "__main__":
    main()
