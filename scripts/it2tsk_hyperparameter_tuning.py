import logging
import sys
import time
import pandas as pd
import numpy as np
import pygad
from itertools import product
import multiprocessing
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
gaussian_config_file_path = "notebooks/experiments/IT2TSK_Gaussian_Exploration.xlsx"


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


def get_gaussian_params() -> dict:
    df = pd.read_excel(gaussian_config_file_path, sheet_name="uf_msr_summary")
    result = {
        item["Sheet Name"]: {
            k: item[k] for k in ["UF_min", "UF_max", "MSR_min", "MSR_max", "R2_max"]
        }
        for item in df.to_dict(orient="records")
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

statistics = []
generation_history = []
best_fitness_history = []
best_params_history = []
all_solutions_history = []
all_fitness_history = []


def explore(
    sheet_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    clusters: Clusters,
    mbk_params: dict,
    gaussian_params: dict,
):
    """
    Efficiently explore IT2TSK Gaussian parameters using GA optimization.

    Key optimizations based on it2tsk_gaussian_exploration:
    1. Reuse pre-computed clusters for all GA evaluations
    2. Pre-scale and pre-transform test data once
    3. Single shared scaler instance for consistency
    4. Minimize redundant operations in fitness function
    """

    logging.info(f"Processing {sheet_name} with {clusters.method.value} clusters...")

    # Create shared scaler that will be reused across all GA evaluations
    shared_scaler = FeatureScaler(decimal_places=4)
    shared_scaler.fit(train_df)
    scaled_train_df = shared_scaler.transform(train_df)

    # Pre-transform test data once (reuse for all predictions)
    scaled_test_df = shared_scaler.transform(test_df)
    X_test_shared = scaled_test_df.drop(columns=[target_column])
    y_test_shared = scaled_test_df[target_column].values

    logging.info("Shared scaler created and data pre-transformed")

    # Define the fitness function
    def fitness_fn(ga_instance, solution, solution_idx):
        try:
            # Create new pipeline instance for each evaluation
            # pipeline = Pipeline(
            #     steps=[
            #         ("feature_scaler", shared_scaler),
            #         ("predictor", IT2TskPredictor(target=target_column)),
            #     ]
            # )

            # Fit pipeline with the solution parameters using pre-scaled data and clusters
            pipeline.fit(
                train_df,  # Use pre-scaled training data
                predictor__clusters=clusters,  # REUSE pre-computed clusters
                predictor__mf_type=MFType2.GAUSSIAN,
                predictor__mf__builder__uncertainty_factor=solution[0],
                predictor__mf__builder__min_std_ratio=solution[1],
                predictor__linear_model=LinearModel.LSE,
            )

            # Use pre-transformed test data for predictions
            y_pred = pipeline.predict(X_test_shared)

            # Calculate accuracy metrics
            r2 = r2_score(y_test_shared, y_pred)
            mse = mean_squared_error(y_test_shared, y_pred)
            mae = mean_absolute_error(y_test_shared, y_pred)

            # Get clustering times
            predictor = pipeline.named_steps["predictor"]
            io_vars = predictor.io_vars_

            total_rule_base_clustering_time = (
                predictor.clusters_.get_execution_time_stats("_create_rules_cluster")[
                    "total_time"
                ]
            )
            total_features_clustering_time = (
                predictor.clusters_.get_execution_time_stats("_create_mfs_clusters")[
                    "total_time"
                ]
            )
            total_clustering_time = (
                total_features_clustering_time + total_rule_base_clustering_time
            )
            average_feature_clustering_time = (
                predictor.clusters_.get_execution_time_stats("_create_mfs_clusters")[
                    "average_time"
                ]
            )
            total_features = len(io_vars.input_variables_) + len(
                io_vars.output_variables_
            )

            # Multiplicative fitness with weight: prioritize both accuracy, error, and speed
            fitness = 0.6 * max(r2, 0) + 0.25 * (1 - mse) + 0.15 * (1 - mae)

            statistics.append(
                {
                    "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Method": clustering_method.value,
                    "Sheet Name": sheet_name,
                    "R2": r2,
                    "MSE": mse,
                    "MAE": mae,
                    "Rule Base Clustering Time": total_rule_base_clustering_time,
                    "Features Clustering Time": total_features_clustering_time,
                    "Clustering Time": total_clustering_time,
                    "Average Feature Clustering Time": average_feature_clustering_time,
                    "Total Features": total_features,
                    "Fitness": fitness,
                    "batch_size": int(mbk_params.get("batch_size")),
                    "tol": mbk_params.get("tol"),
                    "max_no_improvement": int(mbk_params.get("max_no_improvement")),
                    "uncertainty_factor": solution[0],
                    "min_std_ratio": solution[1],
                }
            )

            logging.info(
                f"Sheet: {sheet_name}  R2: {r2:.4f}, MSE: {mse:.4f}, Clustering Time: {total_clustering_time:.2f}s, Fitness: {fitness:.4f}"
            )

            return fitness

        except Exception as e:
            return -1000  # Return very low fitness for failed evaluations

    def on_generation(ga_instance):
        # Store generation number
        generation_history.append(ga_instance.generations_completed)

        # Store best solution and its fitness
        best_solution, best_fitness, _ = ga_instance.best_solution()
        best_fitness_history.append(best_fitness)
        best_params_history.append(best_solution.copy())

        # Store all solutions and their fitness values for this generation
        all_solutions_history.append(ga_instance.population.copy())
        all_fitness_history.append(ga_instance.last_generation_fitness.copy())

    # Define parameter ranges from gaussian_params
    gene_space = [
        {
            "low": gaussian_params.get("UF_min"),
            "high": gaussian_params.get("UF_max"),
            "step": 0.01,
        },  # uncertainty_factor
        {
            "low": gaussian_params.get("MSR_min"),
            "high": gaussian_params.get("MSR_max"),
            "step": 0.01,
        },  # min_std_ratio
    ]

    num_genes = len(gene_space)
    sol_per_pop = 6  # Reduced population size
    num_parents_mating = sol_per_pop // 2
    num_generations = 25  # Reduced for faster execution

    parent_selection_type = "tournament"
    keep_parents = 1  # Keep the best parent at least one
    crossover_type = "uniform"
    mutation_type = "adaptive"
    mutation_num_genes = [2, 1]  # Start with 2 genes (max available), reduce to 1

    logging.info(f"Starting Genetic Algorithm optimization at {sheet_name}...")
    logging.info("Optimizing for best clustering parameters with minimal time")
    logging.info("=" * 60)

    logging.info(f"Create GA instance")
    # Create GA instance
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_fn,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_num_genes=mutation_num_genes,
        on_generation=on_generation,
        stop_criteria=["saturate_5"],  # Stop if no improvement for 5 generations
        parallel_processing=multiprocessing.cpu_count(),
    )

    logging.info(f"Run GA optimization")
    # Run the GA
    ga_instance.run()

    # Get results
    logging.info(f"Get GA solutions")
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    logging.info("Create pipeline with best solution")
    pipeline_best = Pipeline(
        steps=[
            ("feature_scaler", shared_scaler),
            ("predictor", IT2TskPredictor(target=target_column)),
        ]
    )

    # Fit best solution pipeline with pre-scaled data
    pipeline_best.fit(
        train_df,  # Use pre-scaled training data
        predictor__clusters=clusters,  # REUSE pre-computed clusters
        predictor__mf_type=MFType2.GAUSSIAN,
        predictor__mf__builder__uncertainty_factor=solution[0],
        predictor__mf__builder__min_std_ratio=solution[1],
        predictor__linear_model=LinearModel.LSE,
    )

    # Reuse pre-transformed test data for best solution evaluation
    y_pred_best = pipeline_best.predict(X_test_shared)

    r2_best = r2_score(y_test_shared, y_pred_best)
    mse_best = mean_squared_error(y_test_shared, y_pred_best)
    mae_best = mean_absolute_error(y_test_shared, y_pred_best)

    predictor = pipeline.named_steps["predictor"]
    io_vars = predictor.io_vars_

    total_rule_base_clustering_time = predictor.clusters_.get_execution_time_stats(
        "_create_rules_cluster"
    )["total_time"]
    total_features_clustering_time = predictor.clusters_.get_execution_time_stats(
        "_create_mfs_clusters"
    )["total_time"]
    total_clustering_time = (
        total_features_clustering_time + total_rule_base_clustering_time
    )
    average_feature_clustering_time = predictor.clusters_.get_execution_time_stats(
        "_create_mfs_clusters"
    )["average_time"]
    total_features = len(io_vars.input_variables_) + len(io_vars.output_variables_)

    # Multiplicative fitness with weight: prioritize both accuracy, error, and speed
    fitness = 0.6 * max(r2_best, 0) + 0.25 * (1 - mse_best) + 0.15 * (1 - mae_best)

    statistics.append(
        {
            "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Method": clustering_method.value,
            "Sheet Name": sheet_name,
            "R2": r2_best,
            "MSE": mse_best,
            "MAE": mae_best,
            "Rule Base Clustering Time": total_rule_base_clustering_time,
            "Features Clustering Time": total_features_clustering_time,
            "Clustering Time": total_clustering_time,
            "Average Feature Clustering Time": average_feature_clustering_time,
            "Total Features": total_features,
            "Fitness": f"Best Fitness: {fitness}",
            "batch_size": int(mbk_params.get("batch_size")),
            "tol": mbk_params.get("tol"),
            "max_no_improvement": int(mbk_params.get("max_no_improvement")),
            "uncertainty_factor": solution[0],
            "min_std_ratio": solution[1],
        }
    )

    logging.info("\n" + "=" * 60)
    logging.info("OPTIMIZATION COMPLETE")


start = time.time()

# Calculate total work for progress tracking
total_worksheets = len(WorkSheet)
logging.info(f"Starting hyperparameter tuning across {total_worksheets} worksheets")

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
    gaussian_params = get_gaussian_params().get(worksheet.value, {})

    logging.info(f"MBK parameters: {mbk_params}")
    logging.info(
        f"Gaussian parameter ranges: UF=[{gaussian_params.get('UF_min')}-{gaussian_params.get('UF_max')}], MSR=[{gaussian_params.get('MSR_min')}-{gaussian_params.get('MSR_max')}]"
    )

    # OPTIMIZATION 1: Create clusters once per worksheet (not per GA evaluation)
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

    logging.info(f"Pipeline instances created for GA optimization and best solution")

    # OPTIMIZATION 2: Run 10 GA optimizations with pre-computed clusters
    logging.info(f"Starting 10 GA optimization iterations...")

    for i in range(10):
        logging.info(f"--- GA Iteration {i+1}/10 for {worksheet.value} ---")
        explore(
            sheet_name=worksheet.value,
            train_df=train_df,
            test_df=test_df,
            pipeline=pipeline,
            clusters=clusters,  # REUSE pre-computed clusters
            mbk_params=mbk_params,
            gaussian_params=gaussian_params,
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
    "IT2TSK_Hyperparameter_Tuning.xlsx",
    "performances",
)

logging.info("Script finished")
