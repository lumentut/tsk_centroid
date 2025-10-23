from __future__ import annotations


import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .kmeans_cluster_strategy import KMeansClusterStrategy


class MBKMFsCluster(KMeansClusterStrategy):
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
        self.init = init
        self.type = "mbkmeans"
        self.max_k = max_k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.max_no_improvement = max_no_improvement
        self.random_state = random_state

    def fit(self, values: np.ndarray) -> MBKMFsCluster:
        """Fit the MiniBatchKMeans model to the data.

        Args:
            values (np.ndarray): Input data to cluster.

        Returns:
            MBKMFSCluster: Fitted MBKMFSCluster instance.
        """

        values = values.reshape(-1, 1)

        n_clusters = self._calculate_n_clusters(values=values, max_k=self.max_k)

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

        mbkmeans.fit(values)

        centers_ = mbkmeans.cluster_centers_
        # turns into a sorted 1-D copy of the array
        self.centers_ = np.sort(centers_.flatten())
        self.labels_ = mbkmeans.labels_
        self._create_term_names_and_centers(self.centers_, values.min(), values.max())

        return self
