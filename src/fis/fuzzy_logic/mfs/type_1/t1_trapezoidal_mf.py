import numpy as np
from typing import TypedDict
from . import T1BaseMFBuilder, MFType1


class T1TrapezoidalMFParams(TypedDict):
    left: float
    leftTop: float
    rightTop: float
    right: float
    mf: list
    maxHeight: int


class T1TrapezoidalMF(TypedDict):
    name: str
    type: MFType1
    parameters: T1TrapezoidalMFParams
    universe: list


class T1TrapezoidalMFBuilder(T1BaseMFBuilder):
    def __init__(
        self,
        min_std_ratio: float = 0.01,
        support_width_multiplier: float = 1.5,
        decimal_places: int = 4,
    ):
        """Initialize the Type-1 Trapezoidal membership function builder."""
        self.min_std_ratio = min_std_ratio
        self.support_width_multiplier = support_width_multiplier
        self.decimal_places = decimal_places

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> T1TrapezoidalMF:
        """Build a Type-1 Trapezoidal membership function.

        Args:
            column_name (str): The name of the column/feature.
            center (float): The center point of the trapezoidal function.
            values (np.ndarray): All values of the column/feature in the dataset.
            cluster_points (np.ndarray): The data points belonging to this cluster.

        Returns:
            T1TrapezoidalMF: The constructed Type-1 Trapezoidal membership function.
        """
        # Calculate the universe of discourse
        values_min, values_max = values.min(), values.max()

        spread = self.calculate_spread(cluster_points, values)

        width = spread * self.support_width_multiplier
        half_width = width / 2

        # Validate center within bounds
        validated_center = self.validate_center(center, values)

        left = self.rfloat(validated_center - (width * 4))
        left_top = self.rfloat(validated_center - half_width)
        right_top = self.rfloat(validated_center + half_width)
        right = self.rfloat(validated_center + (width * 4))

        # Construct the membership function
        return T1TrapezoidalMF(
            name=column_name,
            type=MFType1.TRAPEZOIDAL,
            parameters=T1TrapezoidalMFParams(
                left=left,
                leftTop=left_top,
                rightTop=right_top,
                right=right,
                mf=[left, left_top, right_top, right],
                maxHeight=1,
            ),
            universe=[self.rfloat(values_min), self.rfloat(values_max)],
        )
