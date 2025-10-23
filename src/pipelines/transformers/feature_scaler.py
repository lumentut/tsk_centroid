import pandas as pd
import numpy as np
from typing import List
from .base_transformer import BaseTransformer


class FeatureScaler(BaseTransformer):
    def __init__(self, cap_range: List[float] = [0.0, 1.0], decimal_places: int = 4):
        """FeatureScaler for scaling numerical features.

        Args:
            decimal_places (int, optional): Number of decimal places to round the scaled values. Defaults to 4.
        """
        super().__init__()  # Initialize BaseTransformer
        self.method = "minmax"
        self.decimal_places = decimal_places
        self.cap_range = cap_range
        self.scaling_stats_ = None
        self.numeric_columns_ = None
        self.fitted_df_ = None
        self.transformed_df_ = None

    def _apply_min_max(self, values: np.ndarray) -> tuple:
        """Min-Max normalization: x' = (x - min(x)) / (max(x) - min(x))"""
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            scaled = np.zeros_like(values)
        else:
            scaled = (values - min_val) / (max_val - min_val)
        stats = {
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val,
            "method": "min_max",
        }
        return scaled, stats

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Fit the scaler to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit

        Returns:
            FeatureScaler: Fitted scaler instance
        """
        if df.equals(self.fitted_df_):
            return self  # Already fitted to this DataFrame

        fitted_df = df.copy()

        # Get numeric columns
        self.numeric_columns_ = self._get_numeric_columns(fitted_df)

        if not self.numeric_columns_:
            print("No numeric columns found for scaling")
            return self

        # Calculate scaling statistics for each numeric column
        self.scaling_stats_ = {}
        for col in self.numeric_columns_:
            values = fitted_df[col].values
            _, stats = self._apply_min_max(values)
            self.scaling_stats_[col] = stats

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame using the fitted scaler.

        Args:
            df (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed DataFrame with scaled features
        """
        if df.equals(self.fitted_df_) and self.transformed_df_ is not None:
            return self.transformed_df_  # Already transformed this DataFrame

        if self.scaling_stats_ is None or self.numeric_columns_ is None:
            raise ValueError("FeatureScaler not fitted yet. Call fit() first.")

        if not self.numeric_columns_:
            return df.copy()

        transformed_df = df.copy()

        for col in self.numeric_columns_:
            if col in transformed_df.columns:
                values = transformed_df[col].values
                stats = self.scaling_stats_[col]

                # Apply scaling using fitted statistics
                if stats["range"] == 0:
                    # Handle constant values
                    scaled_values = np.zeros_like(values)
                else:
                    # Scale using fitted min and range
                    scaled_values = (values - stats["min"]) / stats["range"]
                    # Cap values to [cap_range[0], cap_range[1]] range
                    scaled_values = np.clip(
                        scaled_values, self.cap_range[0], self.cap_range[1]
                    )

                transformed_df[col] = np.round(scaled_values, self.decimal_places)

        self.transformed_df_ = transformed_df

        return transformed_df
