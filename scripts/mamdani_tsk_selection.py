import logging
import sys
import time
import pandas as pd
from datetime import timedelta
from src.dataset import Dataset, WorkSheet
from src.pipelines.predictors import (
    IT2TskPredictor,
    TskPredictor,
    MamdaniPredictor,
    IT2MamdaniPredictor,
)
from src.fis.fuzzy_logic.mfs import MFType2, MFType1
from src.clusters import ClusteringMethod, Clusters
from src.utils.excel_operation import export_to_excel
from scripts.pipelines.fis_pipeline import TSKPipeline, MamdaniPipeline
from src.pipeline import Pipeline
from src.pipelines.transformers import FeatureScaler, Clusterer


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


def predict(
    sheet_name: str,
    clustering_method: ClusteringMethod,
    clusters: Clusters,
    mf_type: str,
    transformer_pipe: Pipeline,
    transformed_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

    logging.info(
        f"Processing {sheet_name} clustering method {clustering_method} mf_type {mf_type} ..."
    )

    if mf_type == "gaussian":
        mf_type_1 = MFType1.GAUSSIAN
        mf_type_2 = MFType2.GAUSSIAN
    elif mf_type == "triangular":
        mf_type_1 = MFType1.TRIANGULAR
        mf_type_2 = MFType2.TRIANGULAR
    elif mf_type == "trapezoidal":
        mf_type_1 = MFType1.TRAPEZOIDAL
        mf_type_2 = MFType2.TRAPEZOIDAL
    else:
        raise ValueError(f"Unsupported mf_type: {mf_type}")

    tsk_pipeline = TSKPipeline(
        fis_type="tsk",
        sheet_name=sheet_name,
        clustering_method=clustering_method,
        transformer_pipe=transformer_pipe,
        transformed_train_df=transformed_train_df,
        test_df=test_df,
        tsk_predictor=TskPredictor(target=target_column),
        clusters=clusters,
        mf_type=mf_type_1,
    )

    statistics.append(tsk_pipeline.statistics)

    it2_tsk_pipeline = TSKPipeline(
        fis_type="it2_tsk",
        sheet_name=sheet_name,
        clustering_method=clustering_method,
        transformer_pipe=transformer_pipe,
        transformed_train_df=transformed_train_df,
        test_df=test_df,
        tsk_predictor=IT2TskPredictor(target=target_column),
        clusters=clusters,
        mf_type=mf_type_2,
    )

    statistics.append(it2_tsk_pipeline.statistics)

    mamdani_pipeline = MamdaniPipeline(
        fis_type="mamdani",
        sheet_name=sheet_name,
        clustering_method=clustering_method,
        transformer_pipe=transformer_pipe,
        transformed_train_df=transformed_train_df,
        test_df=test_df,
        mamdani_predictor=MamdaniPredictor(target=target_column),
        clusters=clusters,
        mf_type=mf_type_1,
    )

    statistics.append(mamdani_pipeline.statistics)

    it2_mamdani_pipeline = MamdaniPipeline(
        fis_type="it2_mamdani",
        sheet_name=sheet_name,
        clustering_method=clustering_method,
        transformer_pipe=transformer_pipe,
        transformed_train_df=transformed_train_df,
        test_df=test_df,
        mamdani_predictor=IT2MamdaniPredictor(target=target_column),
        clusters=clusters,
        mf_type=mf_type_2,
    )

    statistics.append(it2_mamdani_pipeline.statistics)


start = time.time()
for worksheet in WorkSheet:
    dataset = Dataset(path=dataset_path, sheet_name=worksheet.value)
    train_df = dataset.train_df
    test_df = dataset.validate_df

    for clustering_method in [
        ClusteringMethod.KMEANS,
        ClusteringMethod.MBKMEANS,
        ClusteringMethod.FUZZY_C_MEANS,
    ]:
        transformer_pipe = Pipeline(
            steps=[
                ("feature_scaler", FeatureScaler(decimal_places=4)),
                ("clusterer", Clusterer(method=clustering_method)),
            ]
        )

        transformer_pipe.fit(train_df)
        transformed_train_df = transformer_pipe.transform(train_df)
        clusters = transformer_pipe.named_steps["clusterer"].clusters

        for mf_type in ["gaussian", "triangular", "trapezoidal"]:
            predict(
                sheet_name=worksheet.value,
                clustering_method=clustering_method,
                clusters=clusters,
                mf_type=mf_type,
                transformer_pipe=transformer_pipe,
                transformed_train_df=transformed_train_df,
                test_df=test_df,
            )


end = time.time()
elapsed = end - start

logging.info(
    f"Running mamdani vs tsk selection test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    "Mamdani_TSK_Selection.xlsx",
    "fis_selection",
)

logging.info("Script finished")
