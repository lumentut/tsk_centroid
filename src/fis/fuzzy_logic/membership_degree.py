import math
from pyit2fls import trapezoid_mf, tri_mf, gaussian_mf
from src.utils.domain import calculate_range


class T2GaussianMDegree:
    def __init__(self, value: float, **kwargs):
        self.decimal_places: int = kwargs.get("decimalPlaces", 4)
        lmf: list[float] = kwargs.get("lmf", [0.0, 1.0])  # [mean, sigma]
        umf: list[float] = kwargs.get("umf", [0.0, 1.0])
        lmf_max_height: float = kwargs.get("lmfMaxHeight", 1.0)
        umf_max_height: float = kwargs.get("umfMaxHeight", 1.0)

        # Ensure sigma values are not zero to prevent division by zero
        lmf_sigma = max(lmf[1], 1e-6)
        umf_sigma = max(umf[1], 1e-6)

        self.lmf_membership = gaussian_mf(value, [lmf[0], lmf_sigma, lmf_max_height])
        self.umf_membership = gaussian_mf(value, [umf[0], umf_sigma, umf_max_height])

    @property
    def value(self) -> float:
        if hasattr(self, "_md_value"):
            return self._md_value

        # CoG
        self._md_value = round(
            (self.lmf_membership + self.umf_membership) / 2.0, self.decimal_places
        )
        return self._md_value

    def t2_gaussian(self, x, center, sigma, max_height):
        try:
            return max_height * math.exp(-0.5 * ((x - center) / sigma) ** 2)
        except (ZeroDivisionError, OverflowError):
            return 0.0


class T2TriangularMDegree:
    def __init__(self, value: float, **kwargs):
        self.decimal_places: int = kwargs.get("decimalPlaces", 4)
        lmf: list[float] = kwargs.get("lmf", [0.0, 0.0, 0.0])  # [left, peak, right]
        umf: list[float] = kwargs.get("umf", [0.0, 0.0, 0.0])  # [left, peak, right]
        lmf_max_height: float = kwargs.get("lmfMaxHeight", 1.0)
        umf_max_height: float = kwargs.get("umfMaxHeight", 1.0)

        self.lmf_membership = tri_mf(value, [lmf[0], lmf[1], lmf[2], lmf_max_height])
        self.umf_membership = tri_mf(value, [umf[0], umf[1], umf[2], umf_max_height])

    @property
    def value(self) -> float:
        if hasattr(self, "_md_value"):
            return self._md_value

        # CoG
        self._md_value = round(
            (self.lmf_membership + self.umf_membership) / 2.0, self.decimal_places
        )
        return self._md_value


class T2TrapezoidalMDegree:
    def __init__(self, value: float, **kwargs):
        self.decimal_places: int = kwargs.get("decimalPlaces", 4)
        lmf: list[float] = kwargs.get(
            "lmf", [0.0, 0.0, 0.0, 0.0]
        )  # [left, leftTop, rightTop, right]
        umf: list[float] = kwargs.get(
            "umf", [0.0, 0.0, 0.0, 0.0]
        )  # [left, leftTop, rightTop, right]
        lmf_max_height: float = kwargs.get("lmfMaxHeight", 1.0)
        umf_max_height: float = kwargs.get("umfMaxHeight", 1.0)

        self.lmf_membership = trapezoid_mf(
            value, [lmf[0], lmf[1], lmf[2], lmf[3], lmf_max_height]
        )
        self.umf_membership = trapezoid_mf(
            value, [umf[0], umf[1], umf[2], umf[3], umf_max_height]
        )

    @property
    def value(self) -> float:
        if hasattr(self, "_md_value"):
            return self._md_value

        # CoG
        self._md_value = round(
            (self.lmf_membership + self.umf_membership) / 2.0, self.decimal_places
        )
        return self._md_value


class T1TriangularMDegree:
    def __init__(self, value: float, **kwargs):
        left: float = kwargs.get("left", 0.0)
        center: float = kwargs.get("center", left)
        right: float = kwargs.get("right", center)
        max_height: float = kwargs.get("maxHeight", 1.0)

        self._md_value = tri_mf(value, [left, center, right, max_height])

    @property
    def value(self) -> float:
        return self._md_value


class T1GaussianMDegree:
    def __init__(self, value: float, **kwargs):
        mean: float = kwargs.get("mean", 0.0)
        sigma: float = kwargs.get("sigma", 1.0)
        max_height: float = kwargs.get("maxHeight", 1.0)

        # Ensure sigma is not zero to prevent division by zero
        sigma = max(sigma, 1e-6)

        self._md_value = gaussian_mf(value, [mean, sigma, max_height])

    @property
    def value(self) -> float:
        return self._md_value


class T1TrapezoidalMDegree:
    def __init__(self, value: float, **kwargs):
        self.decimal_places: int = kwargs.get("decimalPlaces", 4)
        left: float = kwargs.get("left", 0.0)
        left_top: float = kwargs.get("leftTop", left)
        right_top: float = kwargs.get("rightTop", left_top)
        right: float = kwargs.get("right", right_top)
        max_height: float = kwargs.get("maxHeight", 1.0)

        self._md_value = trapezoid_mf(
            value, [left, left_top, right_top, right, max_height]
        )

    @property
    def value(self) -> float:
        return self._md_value
