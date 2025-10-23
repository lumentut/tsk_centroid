from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .kmeans_cluster_strategy import KMeansClusterStrategy


class KMeansMFsCluster(KMeansClusterStrategy):
    def __init__(
        self,
        init="k-means++",
        max_k=7,
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

    def fit(self, values: np.ndarray) -> KMeansMFsCluster:
        """Fit the KMeans model to the data.

        Args:
            values (np.ndarray): Input data to cluster.

        Returns:
            KmeansMFsCluster: Fitted KmeansMFsCluster instance.
        """

        values = values.reshape(-1, 1)

        n_clusters = self._calculate_n_clusters(values=values, max_k=self.max_k)

        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        kmeans.fit(values)

        centers_ = kmeans.cluster_centers_
        # turns into a sorted 1-D copy of the array
        self.centers_ = np.sort(centers_.flatten())
        self.labels_ = kmeans.labels_
        self._create_term_names_and_centers(self.centers_, values.min(), values.max())

        return self
