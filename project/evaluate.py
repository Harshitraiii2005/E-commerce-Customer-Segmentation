import pandas as pd
import joblib 
import json
import os
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
logging.basicConfig(filename='logs/evaluate.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def evaluate_model(X, labels, model_name):
    results={}
    results['silhouette_score'] = silhouette_score(X, labels)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X, labels) 
    results['davies_bouldin_score'] = davies_bouldin_score(X, labels)

    logging.info(f"{model_name} Evaluation Metrics:")
    logging.info(f"Silhouette Score: {results['silhouette_score']:.4f}")
    logging.info(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    logging.info(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")

    print(f"{model_name} Evaluation Metrics:")
    print(f"Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    print(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")

    return results

def main():
    try:
        df= pd.read_csv("data/clustered_data.csv")
        logging.info(f"Clustered data loaded with shape {df.shape}")

        X=df[['Recency', 'Frequency', 'Monetary']].values


        kmeans = joblib.load('models/kmeans_model.pkl')
        dbscan = joblib.load('models/dbscan_model.pkl')
        Z = joblib.load('models/hierarchical_model.pkl')
        logging.info("Models loaded successfully.")

        metrics = {}

        kmeans_labels = df['KMeans_Cluster'].values
        metrics['KMeans'] = evaluate_model(X, kmeans_labels, "KMeans")

        dbscan_labels = df['DBSCAN_Cluster'].values
        mask = dbscan_labels != -1
        if mask.sum() > 0:
            metrics['DBSCAN'] = evaluate_model(X[mask], dbscan_labels[mask], "DBSCAN")
        else:
            logging.info("DBSCAN found no clusters (all points marked as noise)")

        hier_labels = df['Hierarchical_Cluster'].values
        metrics['Hierarchical'] = evaluate_model(X, hier_labels, "Hierarchical")

        with open('metrics/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Evaluation metrics saved to 'metrics/evaluation_metrics.json'.")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")
        raise e
    
if __name__ == "__main__":
    main()    
