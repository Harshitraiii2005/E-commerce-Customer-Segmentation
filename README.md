E-commerce Customer Segmentation
This project aims to segment e-commerce customers into meaningful groups based on their behavior using unsupervised machine learning techniques.

What I Did
Data Processing & Dimensionality Reduction:
Applied PCA to reduce the original features down to two principal components for easier visualization and efficient modeling.

Clustering Models:
Implemented and compared three clustering algorithms:

KMeans

DBSCAN

Hierarchical Clustering (Agglomerative)

Model Evaluation:
Used multiple clustering evaluation metrics like silhouette score, Calinski-Harabasz index, and Davies-Bouldin score to assess cluster quality and select the best-performing model.

Parameter Tuning & Experiment Tracking:
Leveraged MLPipeline integrated with DVC (Data Version Control) to automate the workflow and track experiments effectively.

Modified params.yaml to tune hyperparameters such as number of clusters, PCA components, and DBSCAN’s eps and min_samples.

Used dvc-live to monitor live metrics during training and evaluation, enabling better decision-making in model selection.

Version Control & Reproducibility:
Managed data, models, and code versions with DVC, ensuring reproducibility and easy collaboration.
This setup also helps in comparing different models and hyperparameter configurations efficiently.

Model Saving:
Saved the final models for later inference and integration with downstream applications.
