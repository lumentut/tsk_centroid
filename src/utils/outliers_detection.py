import pandas as pd
import pydash as _
from typing import TypedDict
from src.utils.pandas_extension import *


class Outlier(TypedDict):
    iqr: list[dict[str, float]]
    zscore: list[dict[str, float]]
    isof: list[dict[str, float]]


def detect_outliers(
    df: pd.DataFrame,
    target_column: str = "TVC",
    skip_columns: list[str] = [],
):
    """Get outlier percentages for each method.
    Args:
        outlier_method (Literal): The outlier detection method to use.
            Options are "outlier_iqr%", "outlier_zscore%", "outlier_isof%".
    Returns:
        dict: Dictionary of outlier percentages for each method.
    """
    descriptive_stats_series_ = df.stats.describe()

    iqr_outlier = (
        descriptive_stats_series_["outlier_iqr%"]
        .drop(index=skip_columns + [target_column])
        .to_dict()
    )

    zscore_outlier = (
        descriptive_stats_series_["outlier_zscore%"]
        .drop(index=skip_columns + [target_column])
        .to_dict()
    )

    isof_outlier = (
        descriptive_stats_series_["outlier_isof%"]
        .drop(index=skip_columns + [target_column])
        .to_dict()
    )

    outlier_iqr_ = _.sort_by(
        [{k: v} for k, v in iqr_outlier.items()],
        lambda o: list(o.values())[0],
    )
    outlier_zscore_ = _.sort_by(
        [{k: v} for k, v in zscore_outlier.items()],
        lambda o: list(o.values())[0],
    )
    outlier_isof_ = _.sort_by(
        [{k: v} for k, v in isof_outlier.items()],
        lambda o: list(o.values())[0],
    )

    return Outlier(iqr=outlier_iqr_, zscore=outlier_zscore_, isof=outlier_isof_)
