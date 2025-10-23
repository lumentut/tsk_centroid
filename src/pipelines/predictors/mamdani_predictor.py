from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
from src.fis.fuzzy_logic import IOVariable, RuleBase, FISType
from src.fis.fuzzy_logic.mfs import MFFactory, MFType1
from src.fis import MamdaniFIS
from src.clusters import Clusters, ClusteringMethod
from .base_predictor import BasePredictor


class MamdaniPredictor(BasePredictor):

    def __init__(self, target: str):
        """Initialize the MamdaniPredictor."""
        self.target = target

    def fit(self, df: pd.DataFrame, **kwargs) -> MamdaniPredictor:
        """Fit the model to the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the training data.

        Returns:
            self: The fitted MamdaniPredictor instance.
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
                name=kwargs.get("mf_type", MFType1.TRIANGULAR), **kwargs
            ),
        )

        self.input_variables_ = self.io_vars_.input_variables_
        self.output_variables_ = self.io_vars_.output_variables_

        self.rule_base_ = RuleBase(
            df=train_df,
            fis_type=FISType.T1_MAMDANI,
            cluster=self.clusters_.rules_cluster_,
            input_variables=self.input_variables_,
            output_variables=self.output_variables_,
        )

        self.model_ = MamdaniFIS(
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
        default_value = 0
        for idx, row in input_df.iterrows():
            input_dict = {col: row[col] for col in input_df.columns}
            try:
                # Suppress division warnings from pyit2fls defuzzification
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        message="invalid value encountered in scalar divide",
                    )
                    output_dict = fis.evaluate(input_dict)

                predicted_value = float(output_dict[1][self.target])

                # Handle NaN or Inf results from defuzzification
                if np.isnan(predicted_value) or np.isinf(predicted_value):
                    predictions.append(default_value)
                else:
                    predictions.append(predicted_value)
            except Exception as e:
                predictions.append(default_value)
                # print(f"Error occurred while evaluating row {idx}: {e}")

        y_pred_ = np.array(predictions)

        return y_pred_
