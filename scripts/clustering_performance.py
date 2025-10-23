import logging
import sys
import time
import pandas as pd
from datetime import timedelta
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.dataset import Dataset, WorkSheet
from src.pipeline import Pipeline
from src.pipelines.transformers import FeatureScaler
from src.pipelines.predictors import IT2TskPredictor
from src.fis.fuzzy_logic.mfs import MFType2
from src.fis.fuzzy_logic.consequents import LinearModel
from src.clusters import ClusteringMethod
from src.utils.excel_operation import export_to_excel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info("========================================")
logging.info("Clustering Comparative Benchmark Test")
logging.info("========================================\n")
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"

target_column = "TVC"

statistics = []


def clustering_performance_time(
    dataset_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method_name: ClusteringMethod,
    iteration_num: int,
):
    logging.info(
        f"Running Dataset name: {dataset_name} Clustering method: {method_name} iteration {iteration_num} ..."
    )

    predictor_pipe = Pipeline(
        steps=[
            ("feature_scaler", FeatureScaler(decimal_places=4)),
            ("predictor", IT2TskPredictor(target=target_column)),
        ]
    )

    predictor_pipe.fit(
        train_df,
        predictor__clustering_method=method_name,
        predictor__mf_type=MFType2.GAUSSIAN,
        predictor__linear_model=LinearModel.LSE,
    )

    transformed_test_df = predictor_pipe.transform(test_df)
    X_test_df = transformed_test_df.drop(columns=[target_column])

    y_test_ = transformed_test_df[target_column].values
    y_pred_ = predictor_pipe.predict(X_test_df)

    r2 = r2_score(y_test_, y_pred_)
    predictor = predictor_pipe.named_steps["predictor"]

    mse = mean_squared_error(y_test_, y_pred_)
    mae = mean_absolute_error(y_test_, y_pred_)

    rule_base = predictor.rule_base_
    io_vars = predictor.io_vars_

    statistics.append(
        {
            "Dataset Name": dataset_name,
            "method": method_name,
            "r2": r2,
            "mse": mse,
            "mae": mae,
            "total_rule_base_clustering_time": predictor.clusters_.get_execution_time_stats(
                "_create_rules_cluster"
            )[
                "total_time"
            ],
            "total_rules": len(rule_base.rules_),
            "total_features_clustering_time": predictor.clusters_.get_execution_time_stats(
                "_create_mfs_clusters"
            )[
                "total_time"
            ],
            "total_features": len(io_vars.input_variables_)
            + len(io_vars.output_variables_),
        }
    )


start = time.time()

logging.info("Running clustering performance test...")

for worksheet in WorkSheet:
    # def predict(threshold: float, sheet_name: str):
    dataset = Dataset(path=dataset_path, sheet_name=worksheet.value)
    train_df = dataset.train_df
    test_df = dataset.validate_df
    for method in [
        ClusteringMethod.MBKMEANS,
        ClusteringMethod.KMEANS,
        ClusteringMethod.FUZZY_C_MEANS,
    ]:
        for i in range(10):
            clustering_performance_time(
                dataset_name=worksheet.value,
                test_df=test_df,
                train_df=train_df,
                method_name=method,
                iteration_num=i,
            )

end = time.time()

elapsed = end - start
logging.info(
    f"Clustering benchmark test completed in {timedelta(seconds=int(elapsed))}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(statistics_df, "Clustering_Performances.xlsx", "clustering_performance")

logging.info("Script finished")
