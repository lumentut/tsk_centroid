from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from .fcm_cluster_strategy import FCMClusterStrategy


class FcmMFsCluster(FCMClusterStrategy):
    def __init__(self, max_k=7, m=1.8, error=0.01, maxiter=1000, random_state=42):
        super().__init__()
        self.type = "fcm"
        self.max_k = max_k
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.random_state = random_state

    def fit(self, values: np.ndarray) -> FcmMFsCluster:
        """Fit the FCM model to the data.

        Args:
            values (np.ndarray): Input data to cluster.

        Returns:
            FcmMFsCluster: Fitted FcmMFsCluster instance.
        """

        values = values.reshape(-1, 1)
        # scikit-fuzzy expects (n_features, n_samples), sklearn uses (n_samples, n_features)
        values = values.T

        n_clusters = self._calculate_n_clusters(values, max_k=self.max_k)

        cluster_centers_, u_, _, _, _, _, _ = fuzz.cluster.cmeans(
            values,
            c=n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter,
            init=None,
            seed=self.random_state,
        )

        centers = cluster_centers_.T

        self.centers_ = np.sort(centers.flatten())
        self.labels_ = np.argmax(u_, axis=0)
        self._create_term_names_and_centers(self.centers_, values.min(), values.max())

        return self
