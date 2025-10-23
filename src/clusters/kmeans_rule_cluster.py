from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .kmeans_cluster_strategy import KMeansClusterStrategy


class KMeansRuleCluster(KMeansClusterStrategy):
    def __init__(
        self,
        init="k-means++",
        max_k=14,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
    ):
        super().__init__()
        self.type = "kmeans"
        self.init = init
        self.max_k = max_k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> KMeansRuleCluster:
        """
        Fit the KMeans clusterer to the input data.
        This method provide learned centers_ & labels_.

        Args:
            df (pd.DataFrame): pandas DataFrame containing the input data to cluster
            Input data to cluster

        Returns:
            self (RulesKMeansClusterer): Fitted clusterer instance
        """

        # Convert DataFrame to numpy array values
        X = df.values
        is_single_column = len(df.columns) == 1

        # Handle different data shapes
        if is_single_column:
            # Single column - reshape to (n_samples, 1) for sklearn
            X = X.reshape(-1, 1)
        # For multiple columns, use the data as-is (rows as samples, columns as features)

        n_clusters = self._calculate_n_clusters(values=X, max_k=self.max_k)

        kmeans = KMeans(
            init=self.init,
            n_clusters=n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        kmeans.fit(X)

        self.labels_ = kmeans.predict(X)
        self.clustered_df_ = df.copy()
        self.clustered_df_["Cluster"] = self.labels_

        self._create_term_names_and_centers(
            cluster_centers_=kmeans.cluster_centers_,
            X=X,
            is_single_column=is_single_column,
        )

        self.centroids_ = np.clip(kmeans.cluster_centers_, 0, 1)
        self.centroids_df_ = pd.DataFrame(self.centroids_, columns=df.columns)

        return self
