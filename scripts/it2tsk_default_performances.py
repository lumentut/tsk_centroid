import logging
import sys
import time
import pandas as pd
from datetime import timedelta
from src.dataset import Dataset, WorkSheet
from src.pipelines.predictors import IT2TskPredictor
from src.fis.fuzzy_logic.mfs import MFType2
from src.utils.excel_operation import export_to_excel
from scripts.pipelines.fis_pipeline import TSKPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info(
    "========================================================================="
)
logging.info("IT2TSK using Gaussian, trapezoidal, and Triangular) Default Performances")
logging.info(
    "=========================================================================\n"
)
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"


target_column = "TVC"

statistics = []


def predict(sheet_name: str):

    logging.info(f"Processing {sheet_name} ...")
    # def predict(threshold: float, sheet_name: str):
    dataset = Dataset(path=dataset_path, sheet_name=sheet_name)
    train_df = dataset.train_df
    test_df = dataset.test_df

    it2_tsk_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        mf_type=MFType2.TRIANGULAR,
    )

    statistics.append(it2_tsk_pipeline.statistics)

    it2_tsk_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        mf_type=MFType2.TRAPEZOIDAL,
    )

    statistics.append(it2_tsk_pipeline.statistics)

    it2_tsk_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        train_df=train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        mf_type=MFType2.GAUSSIAN,
    )

    statistics.append(it2_tsk_pipeline.statistics)


start = time.time()
for worksheet in WorkSheet:
    predict(sheet_name=worksheet.value)


end = time.time()
elapsed = end - start

logging.info(
    f"Running it2tsk performance test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    "it2tsk_default_performances.xlsx",
    "performances",
)

logging.info("Script finished")
