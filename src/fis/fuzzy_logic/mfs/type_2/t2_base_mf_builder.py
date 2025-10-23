import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


class MFType2(Enum):
    GAUSSIAN = "t2_gaussian"
    TRIANGULAR = "t2_triangular"
    TRAPEZOIDAL = "t2_trapezoidal"


class T2BaseMFBuilder(ABC):
    """Base class for membership function builders."""

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> dict:
        """Build a type-2 membership function.

        Args:
            column_name (str): The name of the column.
            center (float): The center of the membership function.
            values (np.ndarray): The values of the column.
            cluster_points (np.ndarray): The points belonging to the cluster.

        Returns:
            dict: The constructed membership function.
        """
        pass

    def rfloat(self, value: float) -> float:
        try:
            return round(float(value), self.decimal_places)
        except (ValueError, TypeError):
            return 0.0

    def calculate_spread(self, cluster_points: np.ndarray, values: np.ndarray) -> float:
        """Calculate the spread/width parameter for the membership function.

        Args:
            cluster_points (np.ndarray): Points belonging to the cluster.
            values (np.ndarray): All values in the dataset.

        Returns:
            float: The calculated spread parameter.
        """
        values_range = values.max() - values.min()
        min_spread = max(
            values_range * self.min_std_ratio, 1e-6
        )  # to avoid division by zero

        if len(cluster_points) > 1:
            cluster_std = np.std(cluster_points)
            return max(cluster_std, min_spread)
        else:
            # Single point cluster - use minimum spread
            return min_spread

    def validate_center(self, center: float, values: np.ndarray) -> float:
        """Validate and potentially adjust the center point.

        Args:
            center (float): The proposed center point.
            values (np.ndarray): All values in the dataset.

        Returns:
            float: The validated center point.
        """
        values_min, values_max = values.min(), values.max()

        # Ensure center is within universe bounds
        validated_center = max(values_min, min(center, values_max))
        return self.rfloat(validated_center)
