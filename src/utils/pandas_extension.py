import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def zscore_percentage(series: pd.Series, threshold: float = 3.0) -> float:
    """
    Calculate outlier percentage using Z-score method.

    Args:
        series: Input pandas Series
        threshold: Z-score threshold (default 3.0, can also use 2.5 or 2.0 for less strict)

    Returns:
        Percentage of outliers
    """
    if len(series) == 0:
        return 0

    # Remove NaN values for calculation
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0

    # Calculate Z-scores
    mean_val = clean_series.mean()
    std_val = clean_series.std()

    # Handle case where standard deviation is 0
    if std_val == 0:
        return 0

    z_scores = np.abs((clean_series - mean_val) / std_val)

    # Identify outliers
    outliers = z_scores > threshold

    return (outliers.sum() / len(clean_series)) * 100


def iqr_percentage(series: pd.Series) -> float:
    """
    Original IQR method for outlier detection.

    Args:
        series: Input pandas Series

    Returns:
        Percentage of outliers
    """
    if len(series) == 0:
        return 0

    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0

    Q1, Q3 = clean_series.quantile([0.25, 0.75])
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (clean_series < lower_bound) | (clean_series > upper_bound)

    return (outliers.sum() / len(clean_series)) * 100


def isof_percentage(series: pd.Series, contamination=0.01, random_state=42) -> float:
    """
    Isolation Forest method for outlier detection.

    Args:
        series: Input pandas Series
        contamination: The proportion of outliers in the data set (default 'auto')
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Percentage of outliers
    """
    if len(series) == 0:
        return 0

    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0

    # Reshape for sklearn (needs 2D array)
    feature_data = clean_series.values.reshape(-1, 1)

    # Initialize and fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,  # You can adjust this
    )

    # Get outlier predictions (-1 for outliers, 1 for inliers)
    outlier_labels = iso_forest.fit_predict(feature_data)

    # Calculate outlier percentage
    outliers_count = np.sum(outlier_labels == -1)

    return (outliers_count / len(clean_series)) * 100


def kurtosis_score(series: pd.Series) -> float:
    """
    Calculate the kurtosis score of a pandas Series.
    """
    if len(series) == 0:
        return 0

    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0

    return clean_series.kurt()


def skewness_score(series: pd.Series) -> float:
    """
    Calculate the skewness score of a pandas Series.
    """
    if len(series) == 0:
        return 0

    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0

    return clean_series.skew()


def coefficient_of_variation(series: pd.Series) -> float:
    return (series.std() / series.mean()) * 100 if series.mean() != 0 else 0


@pd.api.extensions.register_dataframe_accessor("stats")
class CustomStats:
    def __init__(self, pandas_obj):
        """CustomStats constructor

        Args:
            pandas_obj (pd.DataFrame): The pandas DataFrame to extend.
        """
        self._obj = pandas_obj

    def describe(self) -> pd.DataFrame:
        """Generate descriptive statistics + outlier percentage for the DataFrame.

        Raises:
            RuntimeError: If an error occurs while calculating outlier percentages.

        Returns:
            pd.DataFrame: A DataFrame containing the descriptive statistics.
        """
        desc: pd.DataFrame = self._obj.describe().T
        desc["variance"] = self._obj.var()
        desc["skewness"] = self._obj.skew()
        desc["kurtosis"] = self._obj.kurtosis()
        desc["cv"] = self._obj.std() / self._obj.mean()

        try:
            desc["outlier_iqr%"] = self._obj.apply(iqr_percentage)
            desc["outlier_zscore%"] = self._obj.apply(zscore_percentage)
            desc["outlier_isof%"] = self._obj.apply(isof_percentage)

        except Exception as e:
            raise RuntimeError("Error applying outlier percentages calculation") from e

        return desc.round(2)


import pandas as pd
import matplotlib.pyplot as plt


@pd.api.extensions.register_dataframe_accessor("scatter_plot")
class ScatterPlotAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def plot(self, x_col, y_col, **kwargs):
        """
        Create a scatter plot of two features from the DataFrame.

        Args:
            x_col : str
                Column name for x-axis
            y_col : str
                Column name for y-axis
            **kwargs : dict
                Additional keyword arguments to pass to matplotlib scatter plot

        Returns:
            matplotlib.axes.Axes
                The axes object containing the plot
        """
        # Validate that columns exist
        if x_col not in self._obj.columns:
            raise ValueError(f"Column '{x_col}' not found in DataFrame")
        if y_col not in self._obj.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame")

        # Extract default parameters
        figsize = kwargs.pop("figsize", (8, 6))
        title = kwargs.pop("title", f"{y_col} vs {x_col}")
        alpha = kwargs.pop("alpha", 0.7)
        color = kwargs.pop("color", "blue")

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create scatter plot
        ax.scatter(
            self._obj[x_col], self._obj[y_col], alpha=alpha, color=color, **kwargs
        )

        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return ax

    def plot_with_hue(self, x_col, y_col, hue_col=None, **kwargs):
        """
        Create a scatter plot with color coding based on a third column.

        Args:
            x_col : str
                Column name for x-axis
            y_col : str
                Column name for y-axis
            hue_col : str, optional
                Column name for color coding
            **kwargs : dict
                Additional keyword arguments

        Returns:
            matplotlib.axes.Axes
                The axes object containing the plot
        """
        import seaborn as sns

        # Validate columns
        for col in [x_col, y_col]:
            if col not in self._obj.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        if hue_col and hue_col not in self._obj.columns:
            raise ValueError(f"Column '{hue_col}' not found in DataFrame")

        # Extract parameters
        figsize = kwargs.pop("figsize", (10, 6))
        title = kwargs.pop("title", f"{y_col} vs {x_col}")

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        if hue_col:
            # Use seaborn for better color handling
            sns.scatterplot(
                data=self._obj, x=x_col, y=y_col, hue=hue_col, ax=ax, **kwargs
            )
        else:
            ax.scatter(self._obj[x_col], self._obj[y_col], **kwargs)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return ax

    def plot_matrix(self, columns=None, **kwargs):
        """
        Create a scatter plot matrix for multiple features.

        Args:
            columns : list, optional
                List of column names to include. If None, uses all numeric columns.
            **kwargs : dict
                Additional keyword arguments

        Returns:
            matplotlib.figure.Figure
                The figure object containing the plot matrix
        """
        if columns is None:
            # Select only numeric columns
            numeric_cols = self._obj.select_dtypes(include=["number"]).columns.tolist()
            columns = numeric_cols

        # Validate columns
        for col in columns:
            if col not in self._obj.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        # Create scatter plot matrix using pandas
        fig = pd.plotting.scatter_matrix(
            self._obj[columns],
            figsize=kwargs.get("figsize", (12, 12)),
            alpha=kwargs.get("alpha", 0.7),
            diagonal="hist",
        )

        plt.suptitle("Scatter Plot Matrix", y=0.95)
        plt.tight_layout()
        return fig
