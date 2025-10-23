from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod
from src.mixins import PerformanceEvaluationMixin, MfPloterMixin, RuleBaseMixin


class BasePredictor(ABC, PerformanceEvaluationMixin, MfPloterMixin, RuleBaseMixin):
    @abstractmethod
    def fit(self, df: pd.DataFrame, **fit_args) -> BasePredictor:
        """
        Fit the estimator to the data.

        This method should learn any necessary parameters from the input data
        and store them as instance attributes.

        Args:
            df (pd.DataFrame): The input DataFrame to fit the estimator on.
            **fit_args (Any): Additional keyword arguments for fitting.

        Returns:
            BaseEstimator: Returns self to allow method chaining.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict using the fitted estimator.

        This method applies the prediction using parameters learned during fit().
        The estimator must be fitted before calling this method.

        Args:
            df (pd.DataFrame): The input DataFrame to predict.

        Returns:
            pd.Series: The predicted values.

        Raises:
            ValueError: If the estimator is not fitted yet.
        """
        pass
