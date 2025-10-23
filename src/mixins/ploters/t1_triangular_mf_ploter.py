import numpy as np
from .base_mf_ploter import BaseMfPloter


class T1TriangularMfPloter(BaseMfPloter):
    """Type-1 Triangular membership function ploter."""
    
    def _plot_single_membership_function(self, ax, mf: dict, x_range: np.ndarray, color) -> None:
        """Plot a single Type-1 Triangular membership function."""
        params = mf['parameters']
        left = params['left']
        center = params['center']
        right = params['right']
        max_height = params.get('maxHeight', 1.0)
        
        # Calculate triangular membership function values
        y_values = self._calculate_triangular(x_range, left, center, right, max_height)
        
        # Plot the membership function
        ax.plot(x_range, y_values, color=color, linewidth=2, 
                label=f'{mf["name"]}')

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
