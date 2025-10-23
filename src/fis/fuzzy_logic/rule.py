import pandas as pd
import numpy as np
from typing import TypedDict
from src.clusters import ClusterFactory, Cluster, RulesCluster
from src.utils.timing import execution_time, Timing
from .antecedents import Antecedent, Antecedents, LinguisticAntecedent
from .consequents import (
    Consequent,
    Consequents,
    LinguisticConsequent,
    FISType,
    LinearModel,
)
from .io_variable import InputVariable, OutputVariable


class LinguisticRule(TypedDict):
    cluster_index: int
    antecedents: list[LinguisticAntecedent]
    consequents: list[LinguisticConsequent]


class Rule(TypedDict):
    antecedents: list[Antecedent]
    consequents: list[Consequent]


class RuleBase(Timing):
    def __init__(
        self,
        df: pd.DataFrame,
        fis_type: FISType,
        cluster: RulesCluster,
        input_variables: list[InputVariable],
        output_variables: list[OutputVariable],
    ):
        self.input_variables_ = input_variables
        self.output_variables_ = output_variables
        self._rules = None
        self._linguistic_rules = None

        self.cluster = cluster
        self.clusters_data_ = self.create_clusters_data()

        self.antecedents = Antecedents(input_variables)
        self.consequents = Consequents(
            fis_type=fis_type,
            input_vars=input_variables,
            output_vars=output_variables,
            clusters_data_=self.clusters_data_,
            linear_model=LinearModel.LSE,
        )

        self.create_rules(self.cluster.centroids_df_)

    @property
    def rules_(self):
        if self._rules is None:
            return []
        return self._rules

    @property
    def linguistic_rules_(self):
        if self._linguistic_rules is None:
            return []
        return self._linguistic_rules

    @execution_time
    def create_rules(self, centroids_df: pd.DataFrame) -> None:
        """Create fuzzy rules based on the input and output variables."""

        self._rules = []
        self._linguistic_rules = []
        for cluster_idx, clusters in centroids_df.iterrows():

            antecedents, linguistic_antecedents = self.antecedents.create_from_clusters(
                clusters
            )

            consequents, linguistic_consequents = self.consequents.create_from_clusters(
                clusters, cluster_idx
            )

            rule = Rule(
                antecedents=antecedents,
                consequents=consequents,
            )

            linguistic_rule = LinguisticRule(
                cluster_index=cluster_idx,
                antecedents=linguistic_antecedents,
                consequents=linguistic_consequents,
            )

            self._rules.append(rule)
            self._linguistic_rules.append(linguistic_rule)

    @execution_time
    def create_clusters_data(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Get cluster data for a given cluster index.
        """
        clustered_df = self.cluster.clustered_df_.copy()
        clusters = clustered_df["Cluster"].unique()

        clusters_data_ = {}
        for cluster_idx in clusters:
            cluster_df = clustered_df[clustered_df["Cluster"] == cluster_idx]
            cluster_io_df = cluster_df.drop(columns=["Cluster"])

            input_names = [input_var.get("name") for input_var in self.input_variables_]
            output_names = [
                output_var.get("name") for output_var in self.output_variables_
            ]

            X_data = cluster_io_df[input_names].values
            y_data = cluster_io_df[output_names].values
            clusters_data_[cluster_idx] = (X_data, y_data)

        return clusters_data_
