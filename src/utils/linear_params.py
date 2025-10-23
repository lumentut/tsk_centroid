import numpy as np
from sklearn.linear_model import Ridge, LinearRegression


def linear_params_ridge(
    cluster_data: tuple[np.ndarray, np.ndarray],
    # alpha: float = 1e-3
    alpha: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute linear parameters for n-dimensional input using Ridge Regression.

    Args:
    - cluster_data: list of tuples format [([x1, x2, ..., xn], y), ...]
        or [(x1, ..., xn, y), ...]
    - alpha: regularization strength (lambda). Higher = more regularization.

    Returns:
    - coefficients: array of coefficients
    - bias: bias term
    """
    X_data, Y_data = cluster_data

    # Fit Ridge regression
    ridge_model = Ridge(alpha=alpha, fit_intercept=True)  # fit_intercept adds bias term
    ridge_model.fit(X_data, Y_data)

    # Get coefficients and bias
    coefficients = ridge_model.coef_
    bias = ridge_model.intercept_

    return coefficients, bias


def linear_params_lse(
    cluster_data: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute linear parameters for n-dimensional input using Least Squares Estimation.

    Args:
    - cluster_data: list of tuples format [([x1, x2, ..., xn], y), ...]
        or [(x1, ..., xn, y), ...]

    Returns:
    - coefficients: array of coefficients
    - bias: bias term
    """
    X_data, Y_data = cluster_data

    # Fit Linear regression
    lse_model = LinearRegression(fit_intercept=True)
    lse_model.fit(X_data, Y_data)

    # Get coefficients and bias
    coefficients = lse_model.coef_
    bias = lse_model.intercept_

    return coefficients[0], bias
