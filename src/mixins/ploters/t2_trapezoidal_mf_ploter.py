import numpy as np
from .base_mf_ploter import BaseMfPloter


class T2TrapezoidalMfPloter(BaseMfPloter):
    """Type-2 Trapezoidal membership function ploter."""

    def _plot_single_membership_function(
        self, ax, mf: dict, x_range: np.ndarray, color
    ) -> None:
        """Plot a single Type-2 Trapezoidal membership function with its uncertainty footprint."""
        params = mf["parameters"]
        left_lower = params["left_lower"]
        left_top_lower = params["left_top_lower"]
        right_top_lower = params["right_top_lower"]
        right_lower = params["right_lower"]
        left_upper = params["left_upper"]
        left_top_upper = params["left_top_upper"]
        right_top_upper = params["right_top_upper"]
        right_upper = params["right_upper"]

        # Get maximum heights with defaults
        umf_max_height = params.get("umfMaxHeight", 1.0)
        lmf_max_height = params.get("lmfMaxHeight", 1.0)

        # Calculate upper and lower membership functions
        y_upper = self._calculate_trapezoidal(
            x_range,
            left_upper,
            left_top_upper,
            right_top_upper,
            right_upper,
            umf_max_height,
        )
        y_lower = self._calculate_trapezoidal(
            x_range,
            left_lower,
            left_top_lower,
            right_top_lower,
            right_lower,
            lmf_max_height,
        )

        # Plot the footprint of uncertainty (FOU)
        ax.fill_between(
            x_range,
            y_lower,
            y_upper,
            color=color,
            alpha=0.3,
            label=f'{mf["name"]} (FOU)',
        )

        # Plot the boundary functions
        ax.plot(x_range, y_upper, color=color, linewidth=1, linestyle="-")
        ax.plot(x_range, y_lower, color=color, linewidth=1, linestyle="--")

    def _calculate_trapezoidal(
        self,
        x_range: np.ndarray,
        left: float,
        left_top: float,
        right_top: float,
        right: float,
        max_height: float = 1.0,
    ) -> np.ndarray:
        """Calculate trapezoidal membership function values (vectorized)."""
        y_values = np.zeros_like(x_range)

        # Left slope
        mask1 = (x_range >= left) & (x_range < left_top)
        if left_top != left:  # Avoid division by zero
            y_values[mask1] = max_height * (x_range[mask1] - left) / (left_top - left)

        # Flat top
        mask2 = (x_range >= left_top) & (x_range <= right_top)
        y_values[mask2] = max_height

        # Right slope
        mask3 = (x_range > right_top) & (x_range <= right)
        if right != right_top:  # Avoid division by zero
            y_values[mask3] = (
                max_height * (right - x_range[mask3]) / (right - right_top)
            )

        return y_values
