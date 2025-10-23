import numpy as np
from typing import TypedDict
from . import T1BaseMFBuilder, MFType1


class T1GaussianMFParams(TypedDict):
    mean: float
    sigma: float
    mf: list
    maxHeight: int


class T1GaussianMF(TypedDict):
    name: str
    type: MFType1
    parameters: T1GaussianMFParams
    universe: list


class T1GaussianMFBuilder(T1BaseMFBuilder):
    def __init__(self, min_std_ratio: float = 0.01, decimal_places: int = 4):
        """Initialize the Type-1 Gaussian membership function builder."""
        self.min_std_ratio = min_std_ratio
        self.decimal_places = decimal_places

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> T1GaussianMF:
        """Build a Type-1 Gaussian membership function.

        Args:
            column_name (str): The name of the column/feature.
            center (float): The center (mean) of the Gaussian function.
            values (np.ndarray): All values of the column/feature in the dataset.
            cluster_points (np.ndarray): The data points belonging to this cluster.

        Returns:
            dict: Type-1 Gaussian membership function.
        """
        # Calculate universe of discourse
        values_min, values_max = values.min(), values.max()

        # Calculate sigma using the same logic as Type-2 but simpler
        sigma = self.calculate_spread(cluster_points, values)

        # Validate center within bounds
        validated_center = self.validate_center(center, values)

        return T1GaussianMF(
            name=column_name,
            type=MFType1.GAUSSIAN,
            parameters=T1GaussianMFParams(
                mean=validated_center,
                sigma=self.rfloat(sigma),
                mf=[validated_center, self.rfloat(sigma)],  # [mean, sigma] format
                maxHeight=1,
            ),
            universe=[self.rfloat(values_min), self.rfloat(values_max)],
        )
