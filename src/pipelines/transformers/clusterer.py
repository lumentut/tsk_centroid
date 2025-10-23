import pandas as pd
from .base_transformer import BaseTransformer
from src.clusters import Clusters, ClusteringMethod


class Clusterer(BaseTransformer):
    def __init__(self, method: ClusteringMethod, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.transformed_df_ = None
        self.clusters: Clusters = None

    def fit(self, df: pd.DataFrame):
        """No ops fit method"""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a cluster instance to be used as predefined clusters (mfs and rules)
        for efficient time consumption on iterative operations
        that use the same dataframe multiple times.

        Args:
            df (pd.DataFrame): transformed df

        Returns:
            pd.DataFrame: transformed df
        """
        if df.equals(self.transformed_df_) and self.clusters is not None:
            return self.transformed_df_

        transformed_df = df.copy()
        self.clusters = Clusters(df=transformed_df, method=self.method, **self.kwargs)

        return transformed_df
