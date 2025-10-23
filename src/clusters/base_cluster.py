import numpy as np
import pandas as pd
from typing import Union, List
from abc import ABC, abstractmethod

from src.utils.term_name_generator import TermNameGenerator


class BaseCluster(ABC):
    """Abstract base class for all clustering algorithms.

    Args:
        ABC (type): Abstract base class type.
    """

    def __init__(self):
        self.term_name_generator = TermNameGenerator()

    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray]):
        """Fit the clusterer values.

        This method should generate learned centers_ & labels_ to the clusterer instance.

        Args:
            values: The input data to fit.

        Returns:
            BaseCluster: Fitted BaseCluster instance.
        """
        pass

    def _create_term_names_and_centers(
        self, cluster_centers_: np.ndarray, X: np.ndarray, is_single_column: bool
    ) -> None:
        if is_single_column:
            self.centers_ = np.sort(cluster_centers_.flatten())
            self.term_names_ = (
                self.term_name_generator.generate_term_names_from_centers(
                    self.centers_, X.min(), X.max()
                )
            )
        else:
            self.centers_ = cluster_centers_
            first_feature_centers = self.centers_[:, 0]
            sorted_centers = np.sort(first_feature_centers)
            self.term_names_ = (
                self.term_name_generator.generate_term_names_from_centers(
                    sorted_centers, X[:, 0].min(), X[:, 0].max()
                )
            )
