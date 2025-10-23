import logging
import sys
import time
import copy
import pandas as pd
import numpy as np
from itertools import product
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.dataset import Dataset, WorkSheet
from src.pipelines.predictors import IT2TskPredictor
from src.fis.fuzzy_logic.mfs import MFType2
from src.pipelines.transformers.clusterer import Clusterer
from src.utils.excel_operation import export_to_excel
from src.pipeline import Pipeline
from src.clusters.cluster_factory import ClusteringMethod, Clusters
from src.pipelines.transformers.feature_scaler import FeatureScaler
from src.fis.fuzzy_logic.consequents import LinearModel

mbk_config_file_path = "notebooks/experiments/IT2TSK_MBKMeans_Exploration.xlsx"


def get_mbkmeans_params() -> dict:
    df = pd.read_excel(mbk_config_file_path)
    df["Sheet_Order"] = df["Sheet Name"].str.extract(r"^(\d+)").astype(int)
    df_max = df.loc[df.groupby("Sheet Name")["R2"].idxmax()]
    df_max[["R2"]] = df_max[["R2"]].round(4)

    df_sorted = df_max.sort_values("Sheet_Order").reset_index(drop=True)
    selected_max_df = df_sorted[
        ["Sheet Name", "R2", "batch_size", "tol", "max_no_improvement"]
    ]

    selected_max_df.to_dict(orient="records")
    result = {
        item["Sheet Name"]: {
            k: item[k] for k in ["batch_size", "tol", "max_no_improvement"]
        }
        for item in selected_max_df.to_dict(orient="records")
    }
    return result


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info(
    "========================================================================="
)
logging.info("IT2TSK Gaussian MF exploration")
logging.info(
    "=========================================================================\n"
)
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"


target_column = "TVC"
clustering_method = ClusteringMethod.MBKMEANS

statistics = []


def explore(
    sheet_name: str,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clusters: Clusters = None,
    mbk_params: dict = None,
):
    """
    Efficiently explore IT2TSK Gaussian parameters by reusing pre-computed clusters.

    Key optimization: Create one base pipeline and reuse clusters for all parameter combinations.
    This avoids redundant clustering operations while testing different MF builder parameters.
    """

    logging.info(f"Processing {sheet_name} with {clusters.method.value} clusters...")

    logging.info("Test scaler created and data pre-transformed")

    # Parameter grid for exploration
    uncertainty_factors = np.round(np.arange(0.01, 0.3, 0.02), 2)
    min_std_ratios = np.round(np.arange(0.01, 0.18, 0.02), 2)
    param_combinations = list(product(uncertainty_factors, min_std_ratios))

    total_combinations = len(param_combinations)
    logging.info(f"Testing {total_combinations} parameter combinations...")

    # Create shared scaler for efficiency
    shared_scaler = FeatureScaler(decimal_places=4)
    shared_scaler.fit(train_df)

    # Pre-transform test data once for all iterations
    transformed_test_df = shared_scaler.transform(test_df)
    X_test_ = transformed_test_df.drop(columns=[target_column])
    y_test_ = transformed_test_df[target_column].values

    for idx, (uncertainty_factor, min_std_ratio) in enumerate(param_combinations, 1):
        # Fit with pre-computed clusters and current MF parameters
        pipeline.fit(
            train_df,
            predictor__clusters=clusters,  # REUSE pre-computed clusters
            predictor__mf_type=MFType2.GAUSSIAN,
            predictor__linear_model=LinearModel.LSE,
            predictor__mf__builder__uncertainty_factor=uncertainty_factor,
            predictor__mf__builder__min_std_ratio=min_std_ratio,
        )

        y_pred_ = pipeline.predict(X_test_)

        # Calculate accuracy metrics
        r2 = r2_score(y_test_, y_pred_)
        mse = mean_squared_error(y_test_, y_pred_)
        mae = mean_absolute_error(y_test_, y_pred_)

        # Log progress
        if idx % 10 == 0 or idx == total_combinations:
            logging.info(
                f"  Progress: {idx}/{total_combinations} | Current: UF={uncertainty_factor}, MSR={min_std_ratio}, R²={r2:.4f}"
            )

        # Store results
        statistics.append(
            {
                "Sheet Name": sheet_name,
                "Uncertainty Factor": uncertainty_factor,
                "Min Std Ratio": min_std_ratio,
                "R2": r2,
                "MSE": mse,
                "MAE": mae,
                "batch_size": mbk_params.get("batch_size"),
                "tol": mbk_params.get("tol"),
                "max_no_improvement": mbk_params.get("max_no_improvement"),
            }
        )

    logging.info(f"Completed {sheet_name}: {total_combinations} combinations tested")


start = time.time()

# Calculate total work for progress tracking
total_worksheets = len(WorkSheet)
logging.info(f"Starting exploration across {total_worksheets} worksheets")

for worksheet_idx, worksheet in enumerate(WorkSheet, 1):
    worksheet_start = time.time()

    logging.info(f"\n{'='*60}")
    logging.info(f"WORKSHEET {worksheet_idx}/{total_worksheets}: {worksheet.value}")
    logging.info(f"{'='*60}")

    dataset = Dataset(path=dataset_path, sheet_name=worksheet.value)
    train_df = dataset.train_df
    test_df = dataset.validate_df

    logging.info(
        f"Dataset loaded: {train_df.shape[0]} training samples, {test_df.shape[0]} test samples"
    )

    mbk_params = get_mbkmeans_params().get(worksheet.value, {})
    logging.info(f"MBK parameters: {mbk_params}")

    logging.info("Creating clusters (one-time operation per worksheet)...")
    cluster_start = time.time()

    pipeline = Pipeline(
        steps=[
            ("feature_scaler", FeatureScaler(decimal_places=4)),
            ("predictor", IT2TskPredictor(target=target_column)),
        ]
    )

    pipeline.fit(
        train_df,  # pipeline fit only for training dataframe
        predictor__clustering_method=clustering_method,
        predictor__mfs__cluster__batch_size=mbk_params.get("batch_size"),
        predictor__mfs__cluster__tol=mbk_params.get("tol"),
        predictor__mfs__cluster__max_no_improvement=mbk_params.get(
            "max_no_improvement"
        ),
        predictor__rules__cluster__batch_size=mbk_params.get("batch_size"),
        predictor__rules__cluster__tol=mbk_params.get("tol"),
        predictor__rules__cluster__max_no_improvement=mbk_params.get(
            "max_no_improvement"
        ),
        predictor__mf_type=MFType2.GAUSSIAN,
        predictor__linear_model=LinearModel.LSE,
    )

    clusters = pipeline.named_steps["predictor"].clusters_

    cluster_time = time.time() - cluster_start
    logging.info(f"Clustering completed in {cluster_time:.2f}s")

    # Log cluster information
    n_mfs_clusters = {
        feature: len(clusters.mfs_clusters_[feature].centers_)
        for feature in train_df.drop(columns=[target_column]).columns
    }
    n_rules_clusters = len(clusters.rules_cluster_.centroids_)
    logging.info(f"MFS clusters per feature: {n_mfs_clusters}")
    logging.info(f"Rules clusters: {n_rules_clusters}")

    # OPTIMIZATION 2: Run parameter exploration with pre-computed clusters
    explore(
        sheet_name=worksheet.value,
        train_df=train_df,
        test_df=test_df,
        pipeline=pipeline,
        clusters=clusters,
        mbk_params=mbk_params,
    )

    worksheet_time = time.time() - worksheet_start
    logging.info(f"Worksheet {worksheet.value} completed in {worksheet_time:.2f}s")

    # Progress tracking
    overall_progress = worksheet_idx / total_worksheets * 100
    logging.info(
        f"Overall progress: {overall_progress:.1f}% ({worksheet_idx}/{total_worksheets})"
    )


end = time.time()
elapsed = end - start

logging.info(
    f"Running it2tsk performance test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    "IT2TSK_Gaussian_Exploration.xlsx",
    "performances",
)

logging.info("Script finished")
