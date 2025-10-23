import pandas as pd
from typing import TypedDict
from pyit2fls import T1FS, IT2FS
from .linguistic_term import LinguisticTerm
from .fuzzy_sets import FuzzySets
from .io_variable import InputVariable

Antecedent = tuple[str, T1FS | IT2FS]


class LinguisticAntecedent(TypedDict):
    feature_name: str
    linguistic_term: str


class Antecedents:
    """A class representing the antecedents of a fuzzy rule."""

    def __init__(self, input_vars: list[InputVariable], decimal_places: int = 4):
        self.input_vars = input_vars
        self.linguistic_term = LinguisticTerm(decimal_places=decimal_places)
        self.fuzzy_sets = FuzzySets()

    def _antecedent_fuzzy_set(
        self, feature_name: str, linguistic_term: str
    ) -> T1FS | IT2FS:
        """Get input fuzzy sets by name"""
        input_var = next(
            (var for var in self.input_vars if var["name"] == feature_name), None
        )

        membership_function = next(
            (
                mf
                for mf in input_var["membership_functions"]
                if mf["name"] == linguistic_term
            ),
            None,
        )

        if membership_function is None:
            raise ValueError(
                f"Membership function '{linguistic_term}' \
                not found in input variable '{feature_name}'."
            )

        return self.fuzzy_sets.defined_by_mf(membership_function)

    def create_from_clusters(self, clusters: pd.Series) -> list[Antecedent]:
        antecedents: list[tuple[str, Antecedent]] = []
        linguisticAntecedents: list[LinguisticAntecedent] = []

        for input_var in self.input_vars:
            feature_name = input_var["name"]

            if feature_name not in clusters.index:
                raise ValueError(
                    f"Feature '{feature_name}' not found in clusters."
                    f" Available features: {list(clusters.index)}"
                )

            feature_value = float(clusters[feature_name])
            linguistic_term = self.linguistic_term.get_term(feature_value, input_var)
            fuzzy_set = self._antecedent_fuzzy_set(feature_name, linguistic_term)

            antecedents.append((feature_name, fuzzy_set))

            linguisticAntecedents.append(
                LinguisticAntecedent(
                    feature_name=feature_name,
                    linguistic_term=linguistic_term,
                )
            )

        return antecedents, linguisticAntecedents
