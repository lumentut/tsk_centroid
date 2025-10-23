from __future__ import annotations

from typing import Optional, Union
import numpy as np
from .mfs import MFType1, MFType2
from .io_variable import InputVariable, OutputVariable, MembershipFunction
from .membership_degree import (
    T2TriangularMDegree,
    T2GaussianMDegree,
    T2TrapezoidalMDegree,
    T1GaussianMDegree,
    T1TriangularMDegree,
    T1TrapezoidalMDegree,
)

InputOutputVariable = Union[InputVariable, OutputVariable]


class LinguisticTerm:
    def __init__(self, decimal_places: int = 4):
        self.decimal_places = decimal_places

    def get_term(self, value: float, io_variable: InputOutputVariable) -> str:
        """
        Pick MF name with maximal membership degree for supplied numeric value.
        """
        membership_functions = io_variable.get("membership_functions", [])
        if not membership_functions:
            raise ValueError(
                f"No membership functions defined for variable '{io_variable.get('name')}'."
            )

        best_term: Optional[str] = None
        max_degree: float = -1.0

        for mf in membership_functions:
            degree = self._calculate_membership_degree(value, mf)
            if degree > max_degree:
                max_degree = degree
                best_term = mf.get("name")

        # Fallback should never trigger if list non-empty
        return best_term if best_term is not None else membership_functions[0]["name"]

    def get_coefficients_term(
        self, coefficients: dict[str, float], decimal_places: int = 4
    ) -> str:
        """
        Convert coefficients dictionary to a human-readable string.
        """
        terms = []
        for var, coeff in coefficients.items():
            if var == "const":
                terms.append(f"{round(coeff, decimal_places)}")
            else:
                terms.append(f"{round(coeff, decimal_places)}*{var}")
        return " + ".join(terms)

    def get_nearest_mean(self, value: float, io_variable: InputOutputVariable) -> float:
        """
        Return nearest precomputed MF center to value.
        """
        var_name = io_variable.get("name")
        if var_name not in self.mfs_centers:
            raise KeyError(
                f"No centers found for variable '{var_name}' in mfs_centers."
            )

        centers = self.mfs_centers[var_name]
        idx = int(np.argmin(np.abs(centers - value)))

        return float(centers[idx])

    def _calculate_membership_degree(
        self, value: float, membership_function: MembershipFunction
    ) -> float:
        mf_type = membership_function.get("type")
        params = membership_function.get("parameters", {})

        if mf_type == MFType1.GAUSSIAN:
            md = T1GaussianMDegree(value, **params)
            return md.value

        elif mf_type == MFType1.TRIANGULAR:
            md = T1TriangularMDegree(value, **params)
            return md.value

        elif mf_type == MFType1.TRAPEZOIDAL:
            md = T1TrapezoidalMDegree(value, **params)
            return md.value

        elif mf_type == MFType2.GAUSSIAN:
            md = T2GaussianMDegree(value, **params)
            return md.value

        elif mf_type == MFType2.TRIANGULAR:
            md = T2TriangularMDegree(value, **params)
            return md.value

        elif mf_type == MFType2.TRAPEZOIDAL:
            md = T2TrapezoidalMDegree(value, **params)
            return md.value

        else:
            raise ValueError(f"Unsupported membership function type: {mf_type}")
