import numpy as np
from typing import TypedDict
from . import T2BaseMFBuilder, MFType2


class T2TrapezoidalMFParams(TypedDict):
    left_lower: float
    left_top_lower: float
    right_top_lower: float
    right_lower: float
    left_upper: float
    left_top_upper: float
    right_top_upper: float
    right_upper: float
    lmf: list
    umf: list
    lmfMaxHeight: int
    umfMaxHeight: int
    uncertainty_factor: float


class T2TrapezoidalMF(TypedDict):
    name: str
    type: MFType2
    parameters: T2TrapezoidalMFParams
    universe: list


class T2TrapezoidalMFBuilder(T2BaseMFBuilder):
    def __init__(
        self,
        min_std_ratio: float = 0.01,
        min_width_ratio: float = 0.01,
        uncertainty_factor: float = 0.1,
        top_width_multiplier: float = 4.0,
        support_width_multiplier: float = 6.0,
        decimal_places: int = 4,
    ):
        self.min_std_ratio = min_std_ratio
        self.min_width_ratio = min_width_ratio
        self.uncertainty_factor = uncertainty_factor
        self.top_width_multiplier = top_width_multiplier
        self.support_width_multiplier = support_width_multiplier
        self.decimal_places = decimal_places

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> T2TrapezoidalMF:
        """Build a Type-2 Trapezoidal membership function: UMF wider, LMF narrower around T1."""
        # Universe
        values_min, values_max = values.min(), values.max()

        # Spread & width
        spread = self.calculate_spread(cluster_points, values)
        min_width = (values_max - values_min) * self.min_width_ratio
        width = max(spread, min_width) * self.top_width_multiplier  # top plateau width
        half_width = width / 2
        validated_center = self.validate_center(center, values)

        # Support (shoulders)
        support_width = width * self.support_width_multiplier
        half_support = support_width / 2

        # Base T1 Trapezoid (reference)
        left_t1 = self.rfloat(validated_center - half_support)
        left_top_t1 = self.rfloat(validated_center - half_width)
        right_top_t1 = self.rfloat(validated_center + half_width)
        right_t1 = self.rfloat(validated_center + half_support)

        # UMF (wider)
        umf_width = width * (1 + self.uncertainty_factor)
        umf_support = umf_width * self.support_width_multiplier
        umf_half_width = umf_width / 2
        umf_half_support = umf_support / 2

        left_upper = self.rfloat(validated_center - umf_half_support)
        left_top_upper = self.rfloat(validated_center - umf_half_width)
        right_top_upper = self.rfloat(validated_center + umf_half_width)
        right_upper = self.rfloat(validated_center + umf_half_support)

        # LMF (narrower)
        lmf_width = max(width * (1 - self.uncertainty_factor), min_width)
        lmf_support = lmf_width * self.support_width_multiplier
        lmf_half_width = lmf_width / 2
        lmf_half_support = lmf_support / 2

        left_lower = self.rfloat(validated_center - lmf_half_support)
        left_top_lower = self.rfloat(validated_center - lmf_half_width)
        right_top_lower = self.rfloat(validated_center + lmf_half_width)
        right_lower = self.rfloat(validated_center + lmf_half_support)

        return T2TrapezoidalMF(
            name=column_name,
            type=MFType2.TRAPEZOIDAL,
            parameters=T2TrapezoidalMFParams(
                left_lower=left_lower,
                left_top_lower=left_top_lower,
                right_top_lower=right_top_lower,
                right_lower=right_lower,
                left_upper=left_upper,
                left_top_upper=left_top_upper,
                right_top_upper=right_top_upper,
                right_upper=right_upper,
                lmf=[left_lower, left_top_lower, right_top_lower, right_lower],
                umf=[left_upper, left_top_upper, right_top_upper, right_upper],
                lmfMaxHeight=1,
                umfMaxHeight=1,
                uncertainty_factor=self.uncertainty_factor,
            ),
            universe=[self.rfloat(values_min), self.rfloat(values_max)],
        )
