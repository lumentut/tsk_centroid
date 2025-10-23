from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from src.pipelines.predictors.base_predictor import BasePredictor


class MLPPredictor(BasePredictor):

    def __init__(self, target_column: str):
        """Initialize the Neural Network
        (MLP - Multi Layer Perceptron) Predictor."""
        self.target_column = target_column

    def fit(self, df: pd.DataFrame, **kwargs) -> MLPPredictor:
        """Fit the model to the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the training data.

        Returns:
            self: The fitted TskPredictor instance.
        """
        train_df = df.copy()
        X_train_df = train_df.drop(columns=[self.target_column])
        X_train = X_train_df.values
        y_train = train_df[self.target_column].values

        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="tanh",
            solver="adam",
            max_iter=1000,
            random_state=42,
        )

        mlp.fit(X_train, y_train)

        self.model_ = mlp

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

        X_test = input_df.values

        y_pred_ = self.model_.predict(X_test)

        return y_pred_
