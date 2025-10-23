import numpy as np
from .base_mf_ploter import BaseMfPloter


class T2GaussianMfPloter(BaseMfPloter):
    """Type-2 Gaussian membership function ploter."""
    
    def _plot_single_membership_function(self, ax, mf: dict, x_range: np.ndarray, color) -> None:
        """Plot a single Type-2 Gaussian membership function with its uncertainty footprint."""
        params = mf['parameters']
        mean = params['mean']
        sigma_upper = params['sigma_upper']
        sigma_lower = params['sigma_lower']
        
        # Get maximum heights with defaults
        umf_max_height = params.get('umfMaxHeight', 1.0)
        lmf_max_height = params.get('lmfMaxHeight', 1.0)
        
        # Calculate upper and lower membership functions
        y_upper = self._calculate_gaussian(x_range, mean, sigma_upper, umf_max_height)
        y_lower = self._calculate_gaussian(x_range, mean, sigma_lower, lmf_max_height)
        
        # Plot the footprint of uncertainty (FOU)
        ax.fill_between(x_range, y_lower, y_upper, 
                       color=color, alpha=0.3, label=f'{mf["name"]} (FOU)')
        
        # Plot the boundary functions
        ax.plot(x_range, y_upper, color=color, linewidth=1, linestyle='-')
        ax.plot(x_range, y_lower, color=color, linewidth=1, linestyle='--')

    def _calculate_gaussian(self, x_range: np.ndarray, mean: float, sigma: float, 
                           max_height: float = 1.0) -> np.ndarray:
        """Calculate Gaussian membership function values."""
        return max_height * np.exp(-0.5 * ((x_range - mean) / sigma) ** 2)
