import pandas as pd
from .base_transformer import BaseTransformer


class FeatureSelector(BaseTransformer):
    def __init__(self, target: str = "TVC"):
        """FeatureSelector for selecting features based on outlier scores.

        Args:
            threshold (float, optional): Threshold for selecting features based on outlier scores. Defaults to 3.27.
        """
        self.target = target
        self.selected_features_ = None
        self.correlation_scores_ = None

    def fit(self, df: pd.DataFrame, threshold: float = 0.0) -> "FeatureSelector":
        """Fit the selector to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit
            threshold (float, optional): Threshold for selecting features
                based on correlation scores. Defaults to 0.0.

        Returns:
            FeatureSelector: Fitted selector instance
        """
        self.initial_features_ = [c for c in df.columns if c != self.target]
        correlation_matrix = self._get_correlation_matrix(df)
        self.correlation_scores_ = [
            {feature_name: round(abs(corr_value), 2)}
            for feature_name, corr_value in correlation_matrix.items()
            if abs(corr_value) > threshold
        ]

        self.selected_features_ = [list(d.keys())[0] for d in self.correlation_scores_]
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
            not hasattr(self, "correlation_scores_")
            or not hasattr(self, "selected_features_")
            or self.correlation_scores_ is None
            or self.selected_features_ is None
        ):
            raise ValueError(
                "The model has not been fitted yet. Please call 'fit' before 'transform'."
            )

        transformed_df = df.copy()

        # Validate target
        if self.target not in transformed_df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in the input DataFrame."
            )

        available_features = [
            c for c in self.selected_features_ if c in transformed_df.columns
        ]
        if len(available_features) != len(self.selected_features_):
            raise ValueError(
                "Selected features should match the input DataFrame features."
            )

        return transformed_df[available_features + [self.target]]

    def _get_correlation_matrix(self, df: pd.DataFrame) -> pd.Series:
        """Compute the correlation matrix for the features in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the features.

        Returns:
            pd.Series: Series containing the correlation matrices.
        """
        return (
            df.corr(method="pearson")[self.target]
            .drop(self.target)
            .sort_values(ascending=True)
        )
