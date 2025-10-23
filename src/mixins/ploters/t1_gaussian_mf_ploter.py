import numpy as np
from .base_mf_ploter import BaseMfPloter


class T1GaussianMfPloter(BaseMfPloter):
    """Type-1 Gaussian membership function ploter."""
    
    def _plot_single_membership_function(self, ax, mf: dict, x_range: np.ndarray, color) -> None:
        """Plot a single Type-1 Gaussian membership function."""
        params = mf['parameters']
        mean = params['mean']
        sigma = params['sigma']
        max_height = params.get('maxHeight', 1.0)
        
        # Calculate Gaussian membership function values
        y_values = self._calculate_gaussian(x_range, mean, sigma, max_height)
        
        # Plot the membership function
        ax.plot(x_range, y_values, color=color, linewidth=2, 
                label=f'{mf["name"]}')

    def _calculate_gaussian(self, x_range: np.ndarray, mean: float, sigma: float, 
                           max_height: float = 1.0) -> np.ndarray:
        """Calculate Gaussian membership function values."""
        return max_height * np.exp(-0.5 * ((x_range - mean) / sigma) ** 2)
