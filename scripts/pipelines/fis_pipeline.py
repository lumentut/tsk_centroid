import pandas as pd
from abc import ABC
from typing import Union, Literal
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.pipeline import Pipeline
from src.pipelines.transformers import FeatureScaler, CorrelationScoreSelector
from src.pipelines.predictors import (
    TskPredictor,
    IT2TskPredictor,
    MamdaniPredictor,
    IT2MamdaniPredictor,
)
from src.clusters import Clusters, ClusteringMethod
from src.fis.fuzzy_logic.mfs import MFType1, MFType2
from src.fis.fuzzy_logic.consequents import LinearModel


TSKPredictor = Union[TskPredictor, IT2TskPredictor]


class FISPipeline(ABC):
    @property
    def statistics(self):
        return {
            "Worksheet Name": self.sheet_name,
            "FIS Type": self.fis_type,
            "MFs Type": self.mf_type,
            "Clustering Method": self.clustering_method,
            "Selected Features": len(self.selected_features),
            "R2": self.r2,
            "MSE": self.mse,
            "MAE": self.mae,
            "Selected Feature Names": ", ".join(self.selected_features),
            "total_rule_base_clustering_time": self.predictor.clusters_.get_execution_time_stats(
                "_create_rules_cluster"
            )[
                "total_time"
            ],
            "total_rules": len(self.rule_base.rules_),
            "total_features_clustering_time": self.predictor.clusters_.get_execution_time_stats(
                "_create_mfs_clusters"
            )[
                "total_time"
            ],
            "total_features": len(self.io_vars.input_variables_)
            + len(self.io_vars.output_variables_),
            "Total Training Samples": len(self.transformed_train_df),
            "Total Testing Samples": len(self.test_df),
            "Total Samples": len(self.transformed_train_df) + len(self.test_df),
            "total pipeline fit time": self.pipeline.get_execution_time_stats("fit")[
                "total_time"
            ],
            # "total pipeline transform time": self.pipeline.get_execution_time_stats(
            #     "transform"
            # )["total_time"],
            "total pipeline predict time": self.pipeline.get_execution_time_stats(
                "predict"
            )["total_time"],
        }


class TSKPipeline(FISPipeline):
    def __init__(
        self,
        fis_type: Literal["tsk", "it2_tsk"],
        sheet_name: str,
        clustering_method: ClusteringMethod,
        transformer_pipe: Pipeline,
        transformed_train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tsk_predictor: TSKPredictor,
        clusters: Clusters,
        mf_type: Union[MFType1, MFType2],
        target_column: str = "TVC",
    ):
        self.fis_type = fis_type
        self.mf_type = mf_type
        self.clustering_method = clustering_method
        self.sheet_name = sheet_name
        self.transformer_pipe = transformer_pipe
        self.transformed_train_df = transformed_train_df
        self.test_df = test_df

        self.pipeline = Pipeline(
            steps=[
                ("feature_selection", CorrelationScoreSelector(target=target_column)),
                ("predictor", tsk_predictor),
            ]
        )

        self.pipeline.fit(
            self.transformed_train_df,  # pipeline fit only for training dataframe
            predictor__mf_type=mf_type,
            predictor__linear_model=LinearModel.LSE,
            predictor__clusters=clusters,
        )

        transformed_test_df = self.transformer_pipe.transform(self.test_df)
        X_test_df = transformed_test_df.drop(columns=[target_column])
        y_test = transformed_test_df[target_column].values

        y_pred = self.pipeline.predict(X_test_df)

        self.r2 = r2_score(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)

        self.predictor = self.pipeline.named_steps["predictor"]
        self.rule_base = self.predictor.rule_base_
        self.io_vars = self.predictor.io_vars_

        self.selected_features = self.pipeline.named_steps[
            "feature_selection"
        ].selected_features_


MAMDANIPredictor = Union[MamdaniPredictor, IT2MamdaniPredictor]


class MamdaniPipeline(FISPipeline):
    def __init__(
        self,
        fis_type: Literal["mamdani", "it2_mamdani"],
        sheet_name: str,
        clustering_method: ClusteringMethod,
        transformer_pipe: Pipeline,
        transformed_train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mamdani_predictor: MAMDANIPredictor,
        clusters: Clusters,
        mf_type: Union[MFType1, MFType2],
        target_column: str = "TVC",
    ):
        self.fis_type = fis_type
        self.mf_type = mf_type
        self.sheet_name = sheet_name
        self.clustering_method = clustering_method
        self.transformer_pipe = transformer_pipe
        self.transformed_train_df = transformed_train_df
        self.test_df = test_df

        self.pipeline = Pipeline(
            steps=[
                ("feature_selection", CorrelationScoreSelector(target=target_column)),
                ("predictor", mamdani_predictor),
            ]
        )

        self.pipeline.fit(
            self.transformed_train_df,  # pipeline fit only for training dataframe
            predictor__mf_type=mf_type,
            predictor__clusters=clusters,
        )

        transformed_test_df = self.transformer_pipe.transform(self.test_df)
        X_test_df = transformed_test_df.drop(columns=[target_column])
        y_test = transformed_test_df[target_column].values

        y_pred = self.pipeline.predict(X_test_df)

        self.r2 = r2_score(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)

        self.predictor = self.pipeline.named_steps["predictor"]
        self.rule_base = self.predictor.rule_base_
        self.io_vars = self.predictor.io_vars_

        self.selected_features = self.pipeline.named_steps[
            "feature_selection"
        ].selected_features_
