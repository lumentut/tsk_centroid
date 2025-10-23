from __future__ import annotations

import pandas as pd
import numpy as np
from typing import TypedDict, Union
from src.utils.timing import execution_time, Timing
from .mfs import (
    T1BaseMFBuilder,
    T2BaseMFBuilder,
    T1GaussianMF,
    T2GaussianMF,
    T1TriangularMF,
    T2TriangularMF,
)

MembershipFunction = Union[
    T1GaussianMF,
    T2GaussianMF,
    T1TriangularMF,
    T2TriangularMF,
]


class InputVariable(TypedDict):
    name: str
    universe: list[float]
    membership_functions: list[MembershipFunction]


class OutputVariable(TypedDict):
    name: str
    universe: list[float]
    membership_functions: list[MembershipFunction]


class CenterPoints(TypedDict):
    center: float
    points: list[float]


class IOVariable(Timing):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        cluster: dict,
        mf_builder: Union[T1BaseMFBuilder, T2BaseMFBuilder],
        universe: list[float] = [0.0, 1.0],
    ):
        self.target_col = target_col
        self.cluster = cluster
        self.mf_builder = mf_builder
        self.universe = universe
        self.centers_ = {}
        self.mfs_ = {}
        self.mfs_centers_ = {}
        self.features_ = []
        self.targets_ = []
        self.mfs_: dict[str, list[MembershipFunction]] = {}
        self._input_variables: list[InputVariable] = []
        self._output_variables: list[OutputVariable] = []
        self.populate_io_vars(df)

    @property
    def input_variables_(self) -> list[InputVariable]:
        return self._input_variables

    @property
    def output_variables_(self) -> list[OutputVariable]:
        return self._output_variables

    def _populate_input_var(self, input_name: str) -> None:
        self.features_.append(input_name)
        self._input_variables.append(
            InputVariable(
                name=input_name,
                universe=self.universe,
                membership_functions=self.mfs_[input_name],
            )
        )

    def _populate_output_var(self, output_name: str) -> None:
        self.targets_.append(output_name)
        self._output_variables.append(
            OutputVariable(
                name=output_name,
                universe=self.universe,
                membership_functions=self.mfs_[output_name],
            )
        )

    @execution_time
    def populate_io_vars(self, df: pd.DataFrame) -> IOVariable:
        """Populate input and output variables from the DataFrame."""
        columns = df.columns.tolist()
        for col in columns:
            values = df[col].to_numpy()
            cluster = self.cluster[col]
            self.mfs_centers_[col] = cluster.centers_

            mfs: list[MembershipFunction] = []
            for i, (center, col_name) in enumerate(
                zip(cluster.centers_, cluster.term_names_)
            ):
                cluster_points = values[cluster.labels_ == i]
                self.centers_[col_name] = CenterPoints(
                    center=center, points=cluster_points
                )
                mf = self.mf_builder.build(
                    column_name=col_name,
                    center=center,
                    values=values,
                    cluster_points=cluster_points,
                )
                mfs.append(mf)

            self.mfs_[col] = mfs

            if col != self.target_col:
                self._populate_input_var(col)
            else:
                self._populate_output_var(col)

        return self
