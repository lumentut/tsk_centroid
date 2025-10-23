import pandas as pd

from src.pipelines import CorrelationScoreSelector


def correlation_score(
    df: pd.DataFrame,
    target: str,
    skip_columns: list = [],
) -> tuple[list[dict[str, float]], list[float]]:
    """Calculate the correlation scores of features in the DataFrame with respect to the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target variable.
        target (str): The name of the target variable column.
        skip_columns (list, optional): List of columns to skip during correlation calculation. Defaults to ["Minute", "Label"].

    Returns:
        tuple[list[dict[str, float]], list[float]]: A tuple containing a list of dictionaries with feature names and their correlation scores, and a list of correlation score values.
    """
    correlation_df = df.copy()
    selector = CorrelationScoreSelector(target=target)
    selector.fit(correlation_df.drop(columns=skip_columns, axis=1))
    correlation_scores: list[dict[str, float]] = sorted(
        selector.correlation_scores_, key=lambda x: list(x.values())[0]
    )
    correlation_values: list[float] = [
        list(item.values())[0] for item in correlation_scores
    ]

    return correlation_scores, correlation_values
