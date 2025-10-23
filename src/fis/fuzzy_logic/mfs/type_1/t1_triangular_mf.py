import numpy as np
from typing import TypedDict
from . import T1BaseMFBuilder, MFType1


class T1TriangularMFParams(TypedDict):
    center: float
    left: float
    right: float
    mf: list
    maxHeight: int


class T1TriangularMF(TypedDict):
    name: str
    type: MFType1
    parameters: T1TriangularMFParams
    universe: list


class T1TriangularMFBuilder(T1BaseMFBuilder):
    def __init__(
        self,
        min_std_ratio: float = 0.01,
        support_width_multiplier: float = 12.0,
        decimal_places: int = 4,
    ):
        """Initialize the Type-1 triangular membership function builder."""
        self.min_std_ratio = min_std_ratio
        self.support_width_multiplier = support_width_multiplier
        self.decimal_places = decimal_places

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> T1TriangularMF:
        """Build a Type-1 triangular membership function.

        Args:
            column_name (str): The name of the column/feature.
            center (float): The center (peak) of the triangular function.
            values (np.ndarray): All values of the column/feature in the dataset.
            cluster_points (np.ndarray): The data points belonging to this cluster.

        Returns:
            dict: Type-1 triangular membership function.
        """
        # Calculate universe of discourse
        values_min, values_max = values.min(), values.max()

        # Calculate spread/width for the triangle using base class method
        spread = self.calculate_spread(cluster_points, values)

        # For triangular MF, use 2*spread as total width (similar to Type-2 logic)
        width = spread * self.support_width_multiplier
        half_width = width / 2

        # Validate center within bounds
        validated_center = self.validate_center(center, values)

        # Calculate triangle vertices
        left = validated_center - half_width
        right = validated_center + half_width

        return T1TriangularMF(
            name=column_name,
            type=MFType1.TRIANGULAR,
            parameters=T1TriangularMFParams(
                center=validated_center,
                left=self.rfloat(left),
                right=self.rfloat(right),
                mf=[
                    self.rfloat(left),
                    validated_center,
                    self.rfloat(right),
                ],  # [left, center, right] format
                maxHeight=1,
            ),
            universe=[self.rfloat(values_min), self.rfloat(values_max)],
        )
