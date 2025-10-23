from __future__ import annotations

import pandas as pd
from typing import Literal
from src.utils.outliers_detection import detect_outliers
from .base_transformer import BaseTransformer


class OutlierScoreSelector(BaseTransformer):
    def __init__(
        self, target: str = "TVC", method: Literal["iqr", "zscore", "isof"] = "zscore"
    ):
        """FeatureSelector for selecting features based on outlier scores.

        Args:
            threshold (float, optional): Threshold for selecting features based on outlier scores. Defaults to 3.27.
        """
        self.target = target
        self.method = method
        self.selected_features_ = None
        self.outlier_scores_ = None

    def fit(self, df: pd.DataFrame, threshold: float = 0.0) -> OutlierScoreSelector:
        """Fit the selector to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit
            threshold (float, optional): Threshold for selecting features
                based on outliers scores. Defaults to 0.0.

        Returns:
            OutlierScoreSelector: Fitted selector instance
        """
        self.initial_features_ = [c for c in df.columns if c != self.target]
        outlier = detect_outliers(df)

        self.outlier_scores_ = [
            {feature_name: round(abs(outlier_value), 2)}
            for feature_name, outlier_value in [
                list(f.items())[0] for f in outlier["iqr"]
            ]
            if abs(outlier_value) >= threshold
        ]

        self.selected_features_ = [list(d.keys())[0] for d in self.outlier_scores_]
        self.unselected_features_ = [
            f for f in self.initial_features_ if f not in self.selected_features_
        ]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame using the fitted selector.

        Args:
            df (pd.DataFrame): DataFrame to transform

        Returns:
        """
        if (
            not hasattr(self, "outlier_scores_")
            or not hasattr(self, "selected_features_")
            or self.outlier_scores_ is None
            or self.selected_features_ is None
        ):
            raise ValueError(
                "The model has not been fitted yet. Please call 'fit' before 'transform'."
            )

        transformed_df = df.copy()
        original_columns = transformed_df.columns.to_list()

        # Validate target
        if self.target not in original_columns:
            raise ValueError(
                f"Target column '{self.target}' not found in the input DataFrame."
            )

        # columns order is important. it should follows the original columns order seq.
        selected_columns = [
            c for c in original_columns if c in self.selected_features_ + [self.target]
        ]

        if len(selected_columns) != len(self.selected_features_ + [self.target]):
            raise ValueError(
                "Selected features should match the input DataFrame features."
            )

        return transformed_df[selected_columns]
