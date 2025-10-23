import numpy as np
from .base_mf_ploter import BaseMfPloter


class T2TriangularMfPloter(BaseMfPloter):
    """Type-2 Triangular membership function ploter."""
    
    def _plot_single_membership_function(self, ax, mf: dict, x_range: np.ndarray, color) -> None:
        """Plot a single Type-2 Triangular membership function with its uncertainty footprint."""
        params = mf['parameters']
        center = params['center']
        left_lower = params['left_lower']
        right_lower = params['right_lower']
        left_upper = params['left_upper']
        right_upper = params['right_upper']
        
        # Get maximum heights with defaults
        umf_max_height = params.get('umfMaxHeight', 1.0)
        lmf_max_height = params.get('lmfMaxHeight', 1.0)
        
        # Calculate upper and lower membership functions
        y_upper = self._calculate_triangular(x_range, left_upper, center, right_upper, umf_max_height)
        y_lower = self._calculate_triangular(x_range, left_lower, center, right_lower, lmf_max_height)
        
        # Plot the footprint of uncertainty (FOU)
        ax.fill_between(x_range, y_lower, y_upper, 
                       color=color, alpha=0.3, label=f'{mf["name"]} (FOU)')
        
        # Plot the boundary functions
        ax.plot(x_range, y_upper, color=color, linewidth=1, linestyle='-')
        ax.plot(x_range, y_lower, color=color, linewidth=1, linestyle='--')

    def _calculate_triangular(self, x_range: np.ndarray, left: float, center: float, 
                             right: float, max_height: float = 1.0) -> np.ndarray:
        """Calculate triangular membership function values."""
        y_values = np.zeros_like(x_range)
        
        # Left slope (from left to center)
        left_mask = (x_range >= left) & (x_range <= center)
        if center != left:
            y_values[left_mask] = max_height * (x_range[left_mask] - left) / (center - left)
        
        # Right slope (from center to right)
        right_mask = (x_range >= center) & (x_range <= right)
        if right != center:
            y_values[right_mask] = max_height * (right - x_range[right_mask]) / (right - center)
        
        # Peak at center
        center_mask = x_range == center
        y_values[center_mask] = max_height
        
        return y_values
