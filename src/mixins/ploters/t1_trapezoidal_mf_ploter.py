import numpy as np
from .base_mf_ploter import BaseMfPloter


class T1TrapezoidalMfPloter(BaseMfPloter):
    """Type-1 Trapezoidal membership function ploter."""

    def _plot_single_membership_function(
        self, ax, mf: dict, x_range: np.ndarray, color
    ) -> None:
        """Plot a single Type-1 Trapezoidal membership function."""
        params = mf["parameters"]
        a = params["left"]
        b = params["leftTop"]
        c = params["rightTop"]
        d = params["right"]
        max_height = params.get("maxHeight", 1.0)

        # Calculate Trapezoidal membership function values
        y_values = self._calculate_trapezoidal(x_range, a, b, c, d, max_height)

        # Plot the membership function
        ax.plot(x_range, y_values, color=color, linewidth=2, label=f'{mf["name"]}')

    def _calculate_trapezoidal(
        self,
        x_range: np.ndarray,
        a: float,
        b: float,
        c: float,
        d: float,
        max_height: float = 1.0,
    ) -> np.ndarray:
        """Calculate Trapezoidal membership function values."""
        y_values = np.zeros_like(x_range)

        # Calculate the trapezoidal shape
        for i, x in enumerate(x_range):
            if a <= x < b:
                y_values[i] = max_height * (x - a) / (b - a)
            elif b <= x <= c:
                y_values[i] = max_height
            elif c < x <= d:
                y_values[i] = max_height * (d - x) / (d - c)
            else:
                y_values[i] = 0.0

        return y_values
