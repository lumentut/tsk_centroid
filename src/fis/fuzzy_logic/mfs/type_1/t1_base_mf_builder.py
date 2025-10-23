import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


class MFType1(Enum):
    GAUSSIAN = "t1_gaussian"
    TRIANGULAR = "t1_triangular"
    TRAPEZOIDAL = "t1_trapezoidal"


class T1BaseMFBuilder(ABC):
    """Base class for Type-1 membership function builders."""

    @abstractmethod
    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> dict:
        """Build a Type-1 membership function.

        Args:
            column_name (str): The name of the column/feature.
            center (float): The center point of the membership function.
            values (np.ndarray): All values of the column/feature in the dataset.
            cluster_points (np.ndarray): The data points belonging to this cluster.

        Returns:
            dict: The constructed Type-1 membership function containing:
                - name (str): The column/feature name
                - type (str): The type of membership function (e.g., 'gaussian', 'triangular')
                - parameters (dict): Type-1 specific parameters
                - universe (list): The universe of discourse [min, max]
        """
        pass

    def rfloat(self, value: float) -> float:
        """Round a float value to the specified number of decimal places.

        Args:
            value (float): The value to round.

        Returns:
            float: The rounded value, or 0.0 if conversion fails.
        """
        try:
            return round(float(value), self.decimal_places)
        except (ValueError, TypeError):
            return 0.0

    def calculate_universe(self, values: np.ndarray) -> list:
        """Calculate the universe of discourse from input values.

        Args:
            values (np.ndarray): Input values to determine universe bounds.

        Returns:
            list: [min_value, max_value] rounded to decimal places.
        """
        return [self.rfloat(values.min()), self.rfloat(values.max())]

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
