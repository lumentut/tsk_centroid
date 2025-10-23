import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict, Union
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Union


class EvaluationMetrics(TypedDict):
    R_square: float
    MSE: float
    RMSE: float
    MAE: float


class PerformanceEvaluationMixin:
    def evaluate(
        self, y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> dict[str, float]:
        """
        Calculate all performance metrics at once.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            dict: Dictionary containing all metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        evaluation_metrics = EvaluationMetrics(
            R_square=r2_score(y_true, y_pred),
            MSE=mean_squared_error(y_true, y_pred),
            RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
            MAE=mean_absolute_error(y_true, y_pred),
        )

        return evaluation_metrics

    def plot_true_vs_pred(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        title="True vs Predicted Values",
        figsize=(10, 4),
    ):
        """
        Create a scatter plot comparing true vs predicted values.

        Args:
            y_true (array-like): True values (blue points)
            y_pred (array-like): Predicted values (red points)
            title (str): Plot title
            figsize (tuple): Figure size (width, height)
        """

        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Create x-axis (sample indices)
        x = np.arange(len(y_true))

        # Plot scatter points
        ax.scatter(x, y_true, color="blue", alpha=0.7, label="True Values", s=50)
        ax.scatter(x, y_pred, color="red", alpha=0.7, label="Predicted Values", s=50)

        # Set labels and title
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Values")
        ax.set_title(title, fontsize=24, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Calculate and display metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Add metrics text box
        metrics_text = f"R² = {r2:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}"
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()
