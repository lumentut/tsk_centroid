from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
from src.fis.fuzzy_logic import IOVariable, RuleBase, FISType
from src.fis.fuzzy_logic.mfs import MFFactory, MFType1
from src.fis import TskFIS
from src.clusters import Clusters, ClusteringMethod
from .base_predictor import BasePredictor


class TskPredictor(BasePredictor):

    def __init__(self, target: str):
        """Initialize the TskPredictor."""
        self.target = target

    def fit(self, df: pd.DataFrame, **kwargs) -> TskPredictor:
        """Fit the model to the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the training data.

        Returns:
            self: The fitted TskPredictor instance.
        """
        self.decimal_places = kwargs.get("decimal_places", 4)
        clusters = kwargs.get("clusters", None)

        train_df = df.copy()

        if clusters:
            self.clusters_ = clusters
        else:
            clustering_method = kwargs.get(
                "clustering_method", ClusteringMethod.MBKMEANS
            )
            self.clusters_ = Clusters(df=train_df, method=clustering_method, **kwargs)

        self.io_vars_ = IOVariable(
            df=train_df,
            target_col=self.target,
            cluster=self.clusters_.mfs_clusters_,
            mf_builder=MFFactory.create_mf_builder(
                name=kwargs.get("mf_type", MFType1.GAUSSIAN), **kwargs
            ),
        )

        self.input_variables_ = self.io_vars_.input_variables_
        self.output_variables_ = self.io_vars_.output_variables_

        self.rule_base_ = RuleBase(
            df=train_df,
            fis_type=FISType.T1_TSK,
            cluster=self.clusters_.rules_cluster_,
            input_variables=self.input_variables_,
            output_variables=self.output_variables_,
        )

        self.model_ = TskFIS(
            input_variables=self.input_variables_,
            output_variables=self.output_variables_,
            rules=self.rule_base_.rules_,
        )

        return self

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted estimator (model).

        Args:
            input_df (pd.DataFrame): The DataFrame containing the input data for predictions.

        Returns:
            np.ndarray: The predicted values.
        """
        if self.model_ is None:
            raise ValueError(
                "The model has not been fitted yet. \
                Please call 'fit' before 'predict'."
            )

        fis = self.model_.build()

        predictions = []
        for idx, row in input_df.iterrows():
            input_dict = {col: row[col] for col in input_df.columns}
            try:
                # Suppress division warnings from pyit2fls
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message="invalid value encountered in scalar divide",
                    )
                    output_dict = fis.evaluate(
                        input_dict, params=tuple(input_dict.values())
                    )

                predicted_value = float(output_dict[self.target])

                # Handle NaN or Inf results
                if np.isnan(predicted_value) or np.isinf(predicted_value):
                    predicted_value = 0.0

                predictions.append(predicted_value)
            except Exception as e:
                predictions.append(0.0)
                # print(f"Error occurred while evaluating row {idx}: {e}")

        y_pred_ = np.array(predictions)

        return y_pred_
