import logging
import sys
import time
import pandas as pd
from datetime import timedelta
from src.dataset import Dataset, WorkSheet
from src.pipelines.predictors import (
    IT2TskPredictor,
    TskPredictor,
)
from src.fis.fuzzy_logic.mfs import MFType2, MFType1
from src.utils.excel_operation import export_to_excel
from scripts.pipelines.fis_pipeline import TSKPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info("========================================")
logging.info("Mamdani and TSK Selection")
logging.info("========================================\n")
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"


target_column = "TVC"

statistics = []


def predict(sheet_name: str, iteration_num: int):

    # def predict(threshold: float, sheet_name: str):
    dataset = Dataset(path=dataset_path, sheet_name=sheet_name)
    train_df = dataset.train_df
    test_df = dataset.test_df

    logging.info(f"Processing {sheet_name} iteration {iteration_num} ...")

    tsk_gaus_pipeline = TSKPipeline(
        fis_type="tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=TskPredictor(target=target_column),
        mf_type=MFType1.GAUSSIAN,
    )

    statistics.append(tsk_gaus_pipeline.statistics)

    tsk_tri_pipeline = TSKPipeline(
        fis_type="tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=TskPredictor(target=target_column),
        mf_type=MFType1.TRIANGULAR,
    )

    statistics.append(tsk_tri_pipeline.statistics)

    tsk_trap_pipeline = TSKPipeline(
        fis_type="tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=TskPredictor(target=target_column),
        mf_type=MFType1.TRAPEZOIDAL,
    )

    statistics.append(tsk_trap_pipeline.statistics)

    it2_tsk_gaus_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        mf_type=MFType2.GAUSSIAN,
    )

    statistics.append(it2_tsk_gaus_pipeline.statistics)

    it2_tsk_tri_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        mf_type=MFType2.TRIANGULAR,
    )

    statistics.append(it2_tsk_tri_pipeline.statistics)

    it2_tsk_trap_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        mf_type=MFType2.TRAPEZOIDAL,
    )

    statistics.append(it2_tsk_trap_pipeline.statistics)


start = time.time()
for worksheet in WorkSheet:
    for i in range(1000):
        predict(sheet_name=worksheet.value, iteration_num=i)


end = time.time()
elapsed = end - start

logging.info(
    f"Running tsk selection test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    "results_e-nose_dataset_12_beef_cuts.xlsx",
    "tsk_selection",
)

logging.info("Script finished")
