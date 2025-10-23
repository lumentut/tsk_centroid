import logging
import sys
import time
from datetime import timedelta
import pandas as pd
from src.clusters.cluster_factory import ClusteringMethod
from src.dataset import Dataset, WorkSheet
from src.pipelines.predictors import TskPredictor, IT2TskPredictor
from src.utils.excel_operation import export_to_excel
from src.utils.hyperparameter import get_tuned_params
from scripts.pipelines.generic_pipeline import GenericPipeline, ModelMethod
from scripts.pipelines.ml_predictors import (
    MLPPredictor,
    SVRPredictor,
    KNNPredictor,
    RFPredictor,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info("========================================")
logging.info("Comparation comparation between IT2TSK and ML methods")
logging.info("========================================\n")
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"
file_path = "notebooks/experiments/IT2TSK_Hyperparameter_Tuning.xlsx"

target_column = "TVC"
clustering_method = ClusteringMethod.MBKMEANS

statistics = []


def predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sheet_name: str,
    tuned_params: dict,
    iteration_num: int,
):

    logging.info(f"Processing {sheet_name} iteration {iteration_num}  ...")

    tsk_pipeline = GenericPipeline(
        method=ModelMethod.TSK,
        sheet_name=sheet_name,
        tuned_params=tuned_params,
        train_df=train_df,
        test_df=test_df,
        predictor=TskPredictor(target=target_column),
        target_column=target_column,
    )

    statistics.append(tsk_pipeline.statistics)

    it2_tsk_pipeline = GenericPipeline(
        method=ModelMethod.IT2TSK,
        sheet_name=sheet_name,
        tuned_params=tuned_params,
        train_df=train_df,
        test_df=test_df,
        predictor=IT2TskPredictor(target=target_column),
        target_column=target_column,
    )

    statistics.append(it2_tsk_pipeline.statistics)

    mlp_pipeline = GenericPipeline(
        method=ModelMethod.MLP,
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        predictor=MLPPredictor(target_column=target_column),
        target_column=target_column,
    )

    statistics.append(mlp_pipeline.statistics)

    svr_pipeline = GenericPipeline(
        method=ModelMethod.SVR,
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        predictor=SVRPredictor(target_column=target_column),
        target_column=target_column,
    )

    statistics.append(svr_pipeline.statistics)

    rf_pipeline = GenericPipeline(
        method=ModelMethod.RF,
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        predictor=RFPredictor(target_column=target_column),
        target_column=target_column,
    )

    statistics.append(rf_pipeline.statistics)

    knn_pipeline = GenericPipeline(
        method=ModelMethod.KNN,
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        predictor=KNNPredictor(target_column=target_column),
        target_column=target_column,
    )

    statistics.append(knn_pipeline.statistics)


start = time.time()
for worksheet in WorkSheet:
    # def predict(threshold: float, sheet_name: str):
    dataset = Dataset(
        path=dataset_path,
        sheet_name=worksheet.value,
    )
    train_df = dataset.train_df
    test_df = dataset.test_df

    tuned_params = get_tuned_params()[worksheet.value]

    for i in range(1):
        predict(
            train_df=train_df,
            test_df=test_df,
            sheet_name=worksheet.value,
            tuned_params=tuned_params,
            iteration_num=i,
        )


end = time.time()
elapsed = end - start

logging.info(
    f"Running methods comparison test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    "Method_Comparisons.xlsx",
    "methods_comparison",
)

logging.info("Script finished")
