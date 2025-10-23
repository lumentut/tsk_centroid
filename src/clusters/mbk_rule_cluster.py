from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from .kmeans_cluster_strategy import KMeansClusterStrategy


class MBKRuleCluster(KMeansClusterStrategy):
    def __init__(
        self,
        init="k-means++",
        max_k=7,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        batch_size=256,
        max_no_improvement=10,
        random_state=42,
    ):
        super().__init__()
        self.type = "mbkmeans"
        self.init = init
        self.max_k = max_k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.max_no_improvement = max_no_improvement
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> MBKRuleCluster:
        """
        Fit the MiniBatchKMeans clusterer to the input data.
        This method provide learned centers_ & labels_.

        Args:
            df (pd.DataFrame): pandas DataFrame containing the input data to cluster

        Returns:
            MBKRuleCluster: Fitted MBKRuleCluster instance.
        """

        X = df.values
        is_single_column = len(df.columns) == 1

        if is_single_column:
            X = X.reshape(-1, 1)

        n_clusters = self._calculate_n_clusters(values=X, max_k=self.max_k)

        mbkmeans = MiniBatchKMeans(
            init="k-means++",
            n_init=self.n_init,
            n_clusters=n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            batch_size=self.batch_size,
            max_no_improvement=self.max_no_improvement,
            random_state=self.random_state,
        )

        mbkmeans.fit(X)

        # Store the learned centers and labels
        self.labels_ = mbkmeans.predict(X)
        self.clustered_df_ = df.copy()
        self.clustered_df_["Cluster"] = self.labels_

        self._create_term_names_and_centers(
            cluster_centers_=mbkmeans.cluster_centers_,
            X=X,
            is_single_column=is_single_column,
        )

        self.centroids_ = np.clip(mbkmeans.cluster_centers_, 0, 1)
        self.centroids_df_ = pd.DataFrame(self.centroids_, columns=df.columns)

        return self
