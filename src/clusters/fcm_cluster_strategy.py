import numpy as np
import skfuzzy as fuzz
from .base_cluster import BaseCluster


class FCMClusterStrategy(BaseCluster):
    def _calculate_n_clusters(self, values: np.ndarray, max_k: int) -> int:
        """Calculate the optimal number of clusters for FCM.

        Args:
            values (np.ndarray): Input data for clustering.
        Returns:
            int: Optimal number of clusters.
        """
        # Evaluate FCM for cluster counts from 2 to max_k
        pc_scores = {}
        for k in range(2, max_k + 1):
            _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(
                values,
                c=k,
                m=self.m,
                error=self.error,
                maxiter=self.maxiter,
                init=None,
                seed=self.random_state,
            )
            pc_scores[k] = fpc  # fpc is the partition coefficient

        return max(pc_scores, key=pc_scores.get)
