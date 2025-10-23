import numpy as np
from typing import TypedDict
from . import T2BaseMFBuilder, MFType2


class T2TriangularMFParams(TypedDict):
    center: float
    left_lower: float
    right_lower: float
    left_upper: float
    right_upper: float
    lmf: list
    umf: list
    lmfMaxHeight: int
    umfMaxHeight: int
    uncertainty_factor: float


class T2TriangularMF(TypedDict):
    name: str
    type: MFType2
    parameters: T2TriangularMFParams
    universe: list


class T2TriangularMFBuilder(T2BaseMFBuilder):
    def __init__(
        self,
        min_std_ratio: float = 0.01,
        support_width_multiplier: float = 18.0,
        uncertainty_factor: float = 0.1,
        decimal_places: int = 4,
    ):
        self.min_std_ratio = min_std_ratio
        self.support_width_multiplier = support_width_multiplier
        self.uncertainty_factor = uncertainty_factor
        self.decimal_places = decimal_places

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> T2TriangularMF:
        values_min, values_max = values.min(), values.max()

        # Base spread from cluster
        cluster_spread = self.calculate_spread(cluster_points, values)

        base_width = cluster_spread * self.support_width_multiplier

        # Lower (LMF) triangle (narrower)
        lmf_width = base_width * (1 - self.uncertainty_factor)
        lmf_half = lmf_width / 2
        left_lower = center - lmf_half
        right_lower = center + lmf_half

        # Upper (UMF) triangle (wider)
        umf_width = base_width * (1 + self.uncertainty_factor)
        umf_half = umf_width / 2
        left_upper = center - umf_half
        right_upper = center + umf_half

        return T2TriangularMF(
            name=column_name,
            type=MFType2.TRIANGULAR,
            parameters=T2TriangularMFParams(
                center=self.rfloat(center),
                left_lower=self.rfloat(left_lower),
                right_lower=self.rfloat(right_lower),
                left_upper=self.rfloat(left_upper),
                right_upper=self.rfloat(right_upper),
                lmf=[
                    self.rfloat(left_lower),
                    self.rfloat(center),
                    self.rfloat(right_lower),
                ],
                umf=[
                    self.rfloat(left_upper),
                    self.rfloat(center),
                    self.rfloat(right_upper),
                ],
                lmfMaxHeight=1,
                umfMaxHeight=1,
                uncertainty_factor=self.rfloat(self.uncertainty_factor),
            ),
            universe=[self.rfloat(values_min), self.rfloat(values_max)],
        )
