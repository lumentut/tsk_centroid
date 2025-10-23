import numpy as np
from typing import TypedDict
from . import T2BaseMFBuilder, MFType2


class T2GaussianMFParams(TypedDict):
    mean: float
    sigma: float
    lmf: list
    umf: list
    lmfMaxHeight: int
    umfMaxHeight: int
    uncertainty_factor: float


class T2GaussianMF(TypedDict):
    name: str
    type: MFType2
    parameters: T2GaussianMFParams
    universe: list


class T2GaussianMFBuilder(T2BaseMFBuilder):
    def __init__(
        self,
        min_std_ratio: float = 0.01,
        uncertainty_factor: float = 0.01,
        decimal_places: int = 4,
    ):
        self.min_std_ratio = min_std_ratio
        self.uncertainty_factor = uncertainty_factor
        self.decimal_places = decimal_places

    def build(
        self,
        column_name: str,
        center: float,
        values: np.ndarray,
        cluster_points: np.ndarray,
    ) -> T2GaussianMF:
        values_min, values_max = values.min(), values.max()

        sigma = self.calculate_spread(cluster_points, values)

        # Build UMF & LMF from uncertainty factor
        sigma_upper = sigma * (1 + self.uncertainty_factor)  # wider
        sigma_lower = sigma * (1 - self.uncertainty_factor)  # narrower

        return T2GaussianMF(
            name=column_name,
            type=MFType2.GAUSSIAN,
            parameters=T2GaussianMFParams(
                mean=self.rfloat(center),
                sigma_lower=self.rfloat(sigma_lower),
                sigma_upper=self.rfloat(sigma_upper),
                lmf=[self.rfloat(center), self.rfloat(sigma_lower)],  # LMF = tighter
                umf=[self.rfloat(center), self.rfloat(sigma_upper)],  # UMF = looser
                lmfMaxHeight=1,
                umfMaxHeight=1,
                uncertainty_factor=self.rfloat(self.uncertainty_factor),
            ),
            universe=[self.rfloat(values_min), self.rfloat(values_max)],
        )
