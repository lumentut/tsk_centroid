from __future__ import annotations

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from .fcm_cluster_strategy import FCMClusterStrategy


class FcmRuleCluster(FCMClusterStrategy):
    def __init__(self, max_k=14, m=1.8, error=0.01, maxiter=1000, random_state=42):
        super().__init__()
        self.type = "fcm"
        self.max_k = max_k
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> FcmRuleCluster:
        """
        Fit the fuzzy c-means clustering to the input data.

        Args:
            df (pd.DataFrame): pandas DataFrame containing the input data to cluster
            Input data to cluster

        Returns:
            self (FuzzyCMeansWrapper): Fitted clusterer instance
        """
        # Convert DataFrame to numpy array values
        X = df.values
        is_single_column = len(df.columns) == 1

        # Handle different data shapes
        if is_single_column:
            # Single column - reshape to (n_samples, 1) for sklearn compatibility
            X = X.reshape(-1, 1)

        # scikit-fuzzy expects (n_features, n_samples), sklearn uses (n_samples, n_features)
        X = X.T

        n_clusters = self._calculate_n_clusters(X, max_k=self.max_k)

        cluster_centers_, u_, _, _, _, _, _ = fuzz.cluster.cmeans(
            X,
            c=n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter,
            init=None,
            seed=self.random_state,
        )

        self.labels_ = np.argmax(u_, axis=0)
        self.clustered_df_ = df.copy()
        self.clustered_df_["Cluster"] = self.labels_

        self._create_term_names_and_centers(
            cluster_centers_=cluster_centers_,
            X=X,
            is_single_column=is_single_column,
        )

        self.centroids_ = np.clip(self.centers_, 0, 1)
        self.centroids_df_ = pd.DataFrame(self.centroids_, columns=df.columns)

        return self
