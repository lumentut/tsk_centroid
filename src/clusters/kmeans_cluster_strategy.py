import numpy as np
import pandas as pd
from typing import Any
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from .base_cluster import BaseCluster


class KMeansClusterStrategy(BaseCluster):
    """Abstract base class for all KMeans clustering algorithms.

    Args:
        ABC (type): Abstract base class type.
    """

    def _calculate_n_clusters(self, values: np.ndarray, max_k: int) -> int:
        """Calculate the optimal number of clusters for KMeans.

        Args:
            values (np.ndarray): Input data for clustering.

        Returns:
            int: Optimal number of clusters.
        """
        # Check unique values in X to avoid errors/warning in kmeans clustering
        X_df = pd.DataFrame(values)
        max_clusters = len(X_df.drop_duplicates())
        k_range = range(2, min(max_clusters + 1, max_k))
        silhouette_scores_ = {}

        for k in k_range:
            if self.type == "mbkmeans":
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    init=self.init,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    max_no_improvement=self.max_no_improvement,
                    random_state=self.random_state,
                    batch_size=self.batch_size,
                )
            else:
                kmeans = KMeans(
                    n_clusters=k,
                    init=self.init,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )

            labels = kmeans.fit_predict(values)
            if len(set(labels)) > 1:
                silhouette_scores_[k] = silhouette_score(values, labels)
            else:
                silhouette_scores_[k] = -1.0

        n_cluster_ = max(silhouette_scores_, key=silhouette_scores_.get)

        return n_cluster_
