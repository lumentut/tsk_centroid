import pandas as pd
import numpy as np
from pyit2fls import T1FS, IT2FS
from enum import Enum
from typing import TypedDict
from .linguistic_term import LinguisticTerm
from .fuzzy_sets import FuzzySets
from .io_variable import InputVariable, OutputVariable
from src.utils.linear_params import linear_params_ridge, linear_params_lse

Consequent = tuple[str, dict[str, float] | T1FS | IT2FS]


class LinguisticConsequent(TypedDict):
    target_name: str
    linguistic_term: str


class FISType(Enum):
    T1_MAMDANI = "t1_mamdani"
    T2_MAMDANI = "t2_mamdani"
    T1_TSK = "t1_tsk"
    T2_TSK = "t2_tsk"


class LinearModel(Enum):
    RIDGE = "ridge"
    LSE = "ls_estimation"


class Consequents:
    """A class representing the consequents of a fuzzy rule."""

    def __init__(
        self,
        input_vars: list[InputVariable],
        output_vars: list[OutputVariable],
        fis_type: FISType,
        clusters_data_: dict[int, tuple[np.ndarray, np.ndarray]],
        linear_model: LinearModel = LinearModel.LSE,
        decimal_places: int = 4,
    ):
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.clusters_data_ = clusters_data_
        self.linguistic_term = LinguisticTerm(decimal_places=decimal_places)
        self.fuzzy_sets = FuzzySets()
        self.fis_type = fis_type
        self.linear_model = linear_model

    def _consequent_fuzzy_set(
        self, target_name: str, linguistic_term: str
    ) -> T1FS | IT2FS:
        """Get output fuzzy sets by name"""
        output_var = next(
            (var for var in self.output_vars if var["name"] == target_name), None
        )

        membership_function = next(
            (
                mf
                for mf in output_var["membership_functions"]
                if mf["name"] == linguistic_term
            ),
            None,
        )

        if membership_function is None:
            raise ValueError(
                f"Membership function '{linguistic_term}' \
                not found in output variable '{target_name}'."
            )

        return self.fuzzy_sets.defined_by_mf(membership_function)

    def _consequent_coefficients(
        self, feature_names: list[str], cluster_idx: int
    ) -> dict[str, float]:
        """Get consequent coefficients for TSK rules using Ridge Regression."""
        cluster_data = self.clusters_data_[cluster_idx]
        if self.linear_model == LinearModel.RIDGE:
            coefficients, bias = linear_params_ridge(cluster_data)
        elif self.linear_model == LinearModel.LSE:
            coefficients, bias = linear_params_lse(cluster_data)
        else:
            raise ValueError(f"Unsupported linear model: {self.linear_model}")

        result_dict = {"const": bias[0]}
        for i, feature_name in enumerate(feature_names):
            result_dict[feature_name] = coefficients[i]

        return result_dict

    def _consequent_fn(self, coeffs: dict[str, float]):
        variables = [var for var in coeffs.keys() if var != "const"]

        def dynamic_fn(*args, **kwargs):
            # Handle positional arguments
            if args:
                if len(args) != len(variables):
                    raise ValueError(
                        f"Expected {len(variables)} arguments, got {len(args)}"
                    )
                inputs = dict(zip(variables, args))
            else:
                inputs = kwargs
            result = coeffs.get("const", 0)
            for var, coeff in coeffs.items():
                if var != "const" and var in inputs:
                    result += inputs[var] * coeff
            return result

        return dynamic_fn

    def create_from_clusters(
        self, clusters: pd.Series, cluster_idx: int
    ) -> list[Consequent]:
        consequents: list[tuple[str, Consequent]] = []
        linguisticConsequents: list[dict[str, str]] = []

        for output_var in self.output_vars:
            target_name = output_var["name"]

            if target_name not in clusters.index:
                raise ValueError(
                    f"Feature '{target_name}' not found in clusters."
                    f" Available features: {list(clusters.index)}"
                )

            target_value = float(clusters[target_name])

            if self.fis_type in {FISType.T1_MAMDANI, FISType.T2_MAMDANI}:
                linguistic_term = self.linguistic_term.get_term(
                    target_value, output_var
                )
                fuzzy_set = self._consequent_fuzzy_set(target_name, linguistic_term)
                consequents.append((target_name, fuzzy_set))
            elif self.fis_type in {FISType.T1_TSK, FISType.T2_TSK}:
                feature_names = [var["name"] for var in self.input_vars]
                coefficients = self._consequent_coefficients(feature_names, cluster_idx)
                linguistic_term = self.linguistic_term.get_coefficients_term(
                    coefficients
                )
                if self.fis_type == FISType.T1_TSK:
                    fn = self._consequent_fn(coefficients)
                    consequents.append((target_name, fn))
                elif self.fis_type == FISType.T2_TSK:
                    consequents.append((target_name, coefficients))
            else:
                raise ValueError(f"Unsupported FIS type: {self.fis_type}")

            linguisticConsequents.append(
                LinguisticConsequent(
                    target_name=target_name,
                    linguistic_term=linguistic_term,
                )
            )

        return consequents, linguisticConsequents
