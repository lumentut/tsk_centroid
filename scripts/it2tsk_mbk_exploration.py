import logging
import inspect
import sys
import time
import pandas as pd
from datetime import timedelta
import itertools
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.clusters.cluster_factory import Cluster, ClusterContext, ClusteringMethod
from src.dataset import Dataset, WorkSheet
from src.fis.fuzzy_logic.consequents import LinearModel
from src.pipeline import Pipeline
from src.pipelines.predictors import IT2TskPredictor

from src.fis.fuzzy_logic.mfs import MFType2
from src.pipelines.transformers.feature_scaler import FeatureScaler
from src.utils.excel_operation import export_to_excel
from src.fis.fuzzy_logic.mfs.type_2.t2_gaussian_mf import T2GaussianMFBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info("=======================================================")
logging.info("Mini-batch K-Means Clustering Hyperparameter Tuning Script")
logging.info("========================================================\n")
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"


target_column = "TVC"
clustering_method = ClusteringMethod.MBKMEANS

statistics = []
generation_history = []
best_fitness_history = []
best_params_history = []
all_solutions_history = []
all_fitness_history = []

param_grid = {
    "batch_size": [256, 512, 1024],
    "tol": [1e-5, 1e-4, 1e-3],
    "max_no_improvement": [5, 10, 20],
}

t2_gaussian_mf_defaults = inspect.signature(T2GaussianMFBuilder.__init__).parameters
min_std_ratio = t2_gaussian_mf_defaults["min_std_ratio"].default
uncertainty_factor = t2_gaussian_mf_defaults["uncertainty_factor"].default


def explore_worksheet(
    worksheet_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    param_combinations: list,
):
    """
    Efficiently explore MBK clustering parameters for a single worksheet.

    Optimizations:
    1. Pre-scale data once and reuse
    2. Pre-transform test data once per clustering configuration
    3. Reuse pipeline instances where possible
    4. Better progress tracking and timing
    """

    logging.info(f"Processing worksheet: {worksheet_name}")
    logging.info(
        f"Dataset: {train_df.shape[0]} training samples, {test_df.shape[0]} test samples"
    )

    # Create shared scaler for efficiency
    shared_scaler = FeatureScaler(decimal_places=4)
    shared_scaler.fit(train_df)

    # Pre-transform test data once for all iterations
    transformed_test_df = shared_scaler.transform(test_df)
    X_test_ = transformed_test_df.drop(columns=[target_column])
    y_test_ = transformed_test_df[target_column].values

    logging.info("Shared scaler created and test data pre-transformed")

    total_combinations = len(param_combinations)
    logging.info(f"Testing {total_combinations} MBK parameter combinations...")

    pipeline = Pipeline(
        steps=[
            ("feature_scaler", shared_scaler),  # Use shared scaler
            ("predictor", IT2TskPredictor(target=target_column)),
        ]
    )

    for idx, (batch, tol, max_no_improvement) in enumerate(param_combinations, 1):
        param_start = time.time()

        # Fit with clustering parameters - let predictor create internal clusters
        pipeline.fit(
            train_df,  # Use ORIGINAL training data (pipeline will scale it)
            predictor__clustering_method=clustering_method,
            predictor__mfs__cluster__batch_size=batch,
            predictor__mfs__cluster__tol=tol,
            predictor__mfs__cluster__max_no_improvement=max_no_improvement,
            predictor__rules__cluster__batch_size=batch,
            predictor__rules__cluster__tol=tol,
            predictor__rules__cluster__max_no_improvement=max_no_improvement,
            predictor__mf_type=MFType2.GAUSSIAN,
            predictor__linear_model=LinearModel.LSE,
        )

        cluster_time = time.time() - param_start

        # Test data already pre-transformed, just make predictions
        y_pred_ = pipeline.predict(X_test_)

        prediction_time = time.time() - param_start - cluster_time
        total_param_time = time.time() - param_start

        # Calculate accuracy metrics
        r2_ = r2_score(y_test_, y_pred_)
        mse_ = mean_squared_error(y_test_, y_pred_)
        mae_ = mean_absolute_error(y_test_, y_pred_)

        # Get timing and cluster information
        predictor = pipeline.named_steps["predictor"]
        rule_base = predictor.rule_base_
        io_vars = predictor.io_vars_

        # Safely get execution time stats with defaults
        rules_stats = predictor.clusters_.get_execution_time_stats(
            "_create_rules_cluster"
        )
        mfs_stats = predictor.clusters_.get_execution_time_stats("_create_mfs_clusters")

        if rules_stats is None or mfs_stats is None:
            total_rule_base_clustering_time = 0.0
            total_features_clustering_time = 0.0
            total_clustering_time = cluster_time  # Use measured cluster creation time
            average_feature_clustering_time = 0.0
        else:
            total_rule_base_clustering_time = rules_stats["total_time"]
            total_features_clustering_time = mfs_stats["total_time"]
            total_clustering_time = (
                total_features_clustering_time + total_rule_base_clustering_time
            )
            average_feature_clustering_time = mfs_stats["average_time"]

        total_features = len(io_vars.input_variables_) + len(io_vars.output_variables_)

        logging.info(
            f"    R²={r2_:.4f} | batch={batch}, tol={tol}, max_no_improvement={max_no_improvement} | Cluster_time={cluster_time:.3f}s, Total_time={total_param_time:.3f}s"
        )

        # Log progress
        if idx % 9 == 0 or idx == total_combinations:
            logging.info(
                f"  Progress: {idx}/{total_combinations} | Current batch={batch}"
            )

        # Store results with actual clustering parameters used
        statistics.append(
            {
                "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Method": clustering_method.value,
                "Sheet Name": worksheet_name,
                "R2": r2_,
                "MSE": mse_,
                "MAE": mae_,
                "Rule Base Clustering Time": total_rule_base_clustering_time,
                "Features Clustering Time": total_features_clustering_time,
                "Clustering Time": total_clustering_time,
                "Average Feature Clustering Time": average_feature_clustering_time,
                "Total Features": total_features,
                "batch_size": batch,
                "tol": tol,
                "max_no_improvement": max_no_improvement,
                "uncertainty_factor": uncertainty_factor,
                "min_std_ratio": min_std_ratio,
                "Measured_Cluster_Time": cluster_time,
                "Prediction_Time": prediction_time,
                "Total_Param_Time": total_param_time,
            }
        )

    logging.info(
        f"Completed worksheet {worksheet_name}: {total_combinations} combinations tested"
    )


# Main execution flow
start = time.time()

# Calculate total work for progress tracking
total_worksheets = len(WorkSheet)
param_combinations = list(
    itertools.product(
        param_grid["batch_size"],
        param_grid["tol"],
        param_grid["max_no_improvement"],
    )
)
total_combinations_per_sheet = len(param_combinations)

logging.info(f"Starting MBK exploration across {total_worksheets} worksheets")
logging.info(f"Parameter combinations per worksheet: {total_combinations_per_sheet}")
logging.info(
    f"Total combinations across all worksheets: {total_worksheets * total_combinations_per_sheet}"
)

for worksheet_idx, worksheet in enumerate(WorkSheet, 1):
    worksheet_start = time.time()

    logging.info(f"\n{'='*60}")
    logging.info(f"WORKSHEET {worksheet_idx}/{total_worksheets}: {worksheet.value}")
    logging.info(f"{'='*60}")

    dataset = Dataset(path=dataset_path, sheet_name=worksheet.value)
    train_df = dataset.train_df
    test_df = dataset.validate_df

    # Run optimized exploration for this worksheet
    explore_worksheet(
        worksheet_name=worksheet.value,
        train_df=train_df,
        test_df=test_df,
        param_combinations=param_combinations,
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

logging.info("\n" + "=" * 60)
logging.info("OPTIMIZATION COMPLETE")

logging.info(
    f"Running tsk selection test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    f"IT2TSK_MBKMeans_Exploration.xlsx",
    "grid_search",
)

logging.info("Script finished")
