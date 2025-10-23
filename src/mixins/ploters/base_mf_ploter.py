from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from src.fis.fuzzy_logic import MembershipFunction
from src.utils.domain import calculate_range


class BaseMfPloter(ABC):
    """Abstract base class for membership function plotters."""

    def __init__(self, decimal_places: int = 4):
        """Initialize the plotter with common settings.

        Args:
            decimal_places (int): Number of decimal places for calculations (default: 4)
        """
        self.decimal_places = decimal_places

    def plot_mfs_grid(
        self,
        mfs_data: dict[str, list[MembershipFunction]],
        mfs_per_row: int = 2,
        figsize: tuple[int, int] = (12, 12),
    ) -> None:
        """Plot membership functions in a grid layout.

        Args:
            mfs_data (dict[str, list[MembershipFunction]]): Dictionary of membership functions to plot.
            mfs_per_row (int, optional): Number of plots per row. Defaults to 2.
            figsize (tuple[int, int], optional): Size of the figure. Defaults to (12, 12).
        """
        total_mfs = len(mfs_data.keys())
        rows = (total_mfs + mfs_per_row - 1) // mfs_per_row
        cols = mfs_per_row

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle different subplot configurations
        if total_mfs == 1 and rows == 1 and cols == 1:
            axes = [axes]  # Single plot case
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        else:
            axes = axes.flatten()

        for idx, (feature_name, feature_mfs) in enumerate(mfs_data.items()):
            ax = axes[idx]
            self._setup_plot(ax, feature_name)
            self._plot_membership_functions(ax, feature_mfs)

        # Hide empty subplots
        for idx in range(total_mfs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_mf(
        self,
        mf_data: list[MembershipFunction],
        feature_name: str,
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        """Plot membership functions for a single feature.

        Args:
            mf_data (list[MembershipFunction]): List of membership functions to plot.
            feature_name (str): Name of the feature.
            figsize (tuple[int, int], optional): Size of the figure. Defaults to (12, 6).
        """
        if not mf_data:
            raise ValueError("No membership functions provided")

        _, ax = plt.subplots(1, 1, figsize=figsize)
        self._setup_plot(ax, feature_name)
        self._plot_membership_functions(ax, mf_data)

        plt.tight_layout()
        plt.show()

    def _setup_plot(self, ax, feature_name: str) -> None:
        """Configure the plot axes and styling."""
        ax.set_title(f"Membership Functions for {feature_name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Membership Degree")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

    def _plot_membership_functions(self, ax, mf_data: list[dict]) -> None:
        """Plot all membership functions on the given axes."""
        x_range = self._calculate_x_range()
        colors = self._generate_colors(len(mf_data))

        for i, mf in enumerate(mf_data):
            self._plot_single_membership_function(ax, mf, x_range, colors[i])

        ax.legend()

    def _calculate_x_range(self) -> np.ndarray:
        """Calculate the x-axis range for plotting."""
        return calculate_range(0, 1, self.decimal_places)

    def _generate_colors(self, num_colors: int) -> np.ndarray:
        """Generate a colormap for the membership functions."""
        return plt.cm.tab10(np.linspace(0, 1, num_colors))

    @abstractmethod
    def _plot_single_membership_function(
        self, ax, mf: dict, x_range: np.ndarray, color
    ) -> None:
        """Plot a single membership function.

        Args:
            ax: Matplotlib axes object
            mf (dict): Membership function parameters
            x_range (np.ndarray): X-axis range for plotting
            color: Color for the plot
        """
        pass
