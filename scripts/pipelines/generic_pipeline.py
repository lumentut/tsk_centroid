import pandas as pd
from enum import Enum
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.pipeline import Pipeline
from src.pipelines.transformers import FeatureScaler
from src.pipelines.predictors.base_predictor import BasePredictor
from src.clusters import ClusteringMethod
from src.fis.fuzzy_logic.mfs import MFType1, MFType2
from src.fis.fuzzy_logic.consequents import LinearModel


class ModelMethod(Enum):
    TSK = ("tsk_fls",)
    IT2TSK = ("it2_tsk_fls",)
    MLP = ("mlp_neural_network",)
    SVR = ("support_vector_regression",)
    RF = ("random_forest",)
    KNN = ("k_nearest_neighbors",)


class GenericPipeline:
    def __init__(
        self,
        method: ModelMethod,
        sheet_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictor: BasePredictor,
        tuned_params: dict = {},
        target_column: str = "TVC",
    ):
        self.method = method
        self.sheet_name = sheet_name
        self.target_column = target_column
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.tuned_params = tuned_params
        self.pipeline = Pipeline(
            steps=[
                ("feature_scaler", FeatureScaler(decimal_places=4)),
                ("predictor", predictor),
            ]
        )

        if self.method == ModelMethod.IT2TSK:
            self.pipeline.fit(
                self.train_df,  # pipeline fit only for training dataframe
                predictor__clustering_method=ClusteringMethod.MBKMEANS,
                predictor__mfs__cluster__batch_size=tuned_params.get("batch_size"),
                predictor__mfs__cluster__tol=tuned_params.get("tol"),
                predictor__mfs__cluster__max_no_improvement=tuned_params.get(
                    "max_no_improvement"
                ),  # Fixed for simplicity
                predictor__rules__cluster__batch_size=tuned_params.get("batch_size"),
                predictor__rules__cluster__tol=tuned_params.get("tol"),
                predictor__rules__cluster__max_no_improvement=tuned_params.get(
                    "max_no_improvement"
                ),  # Fixed for simplicity
                predictor__mf_type=MFType2.GAUSSIAN,
                predictor__linear_model=LinearModel.LSE,
                predictor__mf__builder__uncertainty_factor=tuned_params.get(
                    "uncertainty_factor"
                ),
                predictor__mf__builder__min_std_ratio=tuned_params.get("min_std_ratio"),
            )

        elif self.method == ModelMethod.TSK:
            self.pipeline.fit(
                self.train_df,  # pipeline fit only for training dataframe
                predictor__clustering_method=ClusteringMethod.MBKMEANS,
                predictor__mfs__cluster__batch_size=tuned_params.get("batch_size"),
                predictor__mfs__cluster__tol=tuned_params.get("tol"),
                predictor__mfs__cluster__max_no_improvement=tuned_params.get(
                    "max_no_improvement"
                ),  # Fixed for simplicity
                predictor__rules__cluster__batch_size=tuned_params.get("batch_size"),
                predictor__rules__cluster__tol=tuned_params.get("tol"),
                predictor__rules__cluster__max_no_improvement=tuned_params.get(
                    "max_no_improvement"
                ),  # Fixed for simplicity
                predictor__mf_type=MFType1.GAUSSIAN,
                predictor__linear_model=LinearModel.LSE,
            )
        else:
            self.pipeline.fit(self.train_df)

        transformed_test_df = self.pipeline.transform(self.test_df)
        X_test_df = transformed_test_df.drop(columns=[target_column])

        y_test_ = transformed_test_df[target_column].values
        y_pred_ = self.pipeline.predict(X_test_df)

        self.r2 = r2_score(y_test_, y_pred_)
        self.mse = mean_squared_error(y_test_, y_pred_)
        self.mae = mean_absolute_error(y_test_, y_pred_)

    @property
    def statistics(self):
        return {
            "Worksheet Name": self.sheet_name,
            "Predictor Method": self.method,
            "R2": self.r2,
            "MSE": self.mse,
            "MAE": self.mae,
            "Total Training Samples": len(self.train_df),
            "Total Testing Samples": len(self.test_df),
            "Total Samples": len(self.train_df) + len(self.test_df),
            "total pipeline fit time": self.pipeline.get_execution_time_stats("fit")[
                "total_time"
            ],
            "total pipeline transform time": self.pipeline.get_execution_time_stats(
                "transform"
            )["total_time"],
            "total pipeline predict time": self.pipeline.get_execution_time_stats(
                "predict"
            )["total_time"],
        }
