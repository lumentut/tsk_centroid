import numpy as np


def calculate_num_points(domain_min: int, domain_max: int, decimal_places: int):
    """
    Calculate the number of points needed for np.linspace based on decimal places.

    Parameters:
    domain_min (float): Minimum value of the domain
    domain_max (float): Maximum value of the domain
    decimal_places (int): Number of decimal places (1->0.1, 2->0.01, 3->0.001, etc.)

    Returns:
    int: Number of points needed for the desired step size
    """
    # Calculate step size from decimal places
    step_size = 10 ** (-decimal_places)

    # Calculate number of points
    num_points = int((domain_max - domain_min) / step_size) + 1

    return num_points


def calculate_range(domain_min: int, domain_max: int, decimal_places: int):
    """
    Calculate the range based on decimal places.

    Parameters:
    domain_min (float): Minimum value of the domain
    domain_max (float): Maximum value of the domain
    decimal_places (int): Number of decimal places (1->0.1, 2->0.01, 3->0.001, etc.)

    Returns:
    range (np.array): Array of range
    """
    num_points = calculate_num_points(domain_min, domain_max, decimal_places)

    x_range = np.linspace(domain_min, domain_max, num_points)
    x_range[0] = domain_min  # Ensure starts at domain_min
    x_range[-1] = domain_max  # Ensure ends at domain_max

    return x_range
