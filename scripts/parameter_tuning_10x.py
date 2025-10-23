import logging
import sys
import time
import pandas as pd
from datetime import timedelta
import pygad
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.clusters.cluster_factory import Cluster, ClusterContext, ClusteringMethod
from src.dataset import Dataset, WorkSheet
from src.fis.fuzzy_logic.consequents import LinearModel
from src.pipeline import Pipeline
from src.pipelines.predictors import IT2TskPredictor

from src.fis.fuzzy_logic.mfs import MFType2
from src.pipelines.transformers.feature_scaler import FeatureScaler
from src.utils.excel_operation import export_to_excel


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

gene_spaces = {}

gene_spaces[WorkSheet.DS1.value] = [
    {"low": 0.05, "high": 0.17},  # UF
    {"low": 0.01, "high": 0.15},  # MSR
]

gene_spaces[WorkSheet.DS2.value] = [
    {"low": 0.05, "high": 0.26},
    {"low": 0.03, "high": 0.15},
]

gene_spaces[WorkSheet.DS3.value] = [
    {"low": 0.05, "high": 0.19},
    {"low": 0.01, "high": 0.11},
]

gene_spaces[WorkSheet.DS4.value] = [
    {"low": 0.05, "high": 0.28},
    {"low": 0.01, "high": 0.15},
]

gene_spaces[WorkSheet.DS5.value] = [
    {"low": 0.05, "high": 0.35},
    {"low": 0.01, "high": 0.09},
]

gene_spaces[WorkSheet.DS6.value] = [
    {"low": 0.05, "high": 0.23},
    {"low": 0.05, "high": 0.15},
]

gene_spaces[WorkSheet.DS7.value] = [
    {"low": 0.05, "high": 0.16},
    {"low": 0.01, "high": 0.15},
]

gene_spaces[WorkSheet.DS8.value] = [
    {"low": 0.05, "high": 0.22},
    {"low": 0.01, "high": 0.15},
]

gene_spaces[WorkSheet.DS9.value] = [
    {"low": 0.05, "high": 0.23},
    {"low": 0.01, "high": 0.08},
]

gene_spaces[WorkSheet.DS10.value] = [
    {"low": 0.05, "high": 0.26},
    {"low": 0.03, "high": 0.15},
]
gene_spaces[WorkSheet.DS11.value] = [
    {"low": 0.05, "high": 0.19},
    {"low": 0.01, "high": 0.15},
]

gene_spaces[WorkSheet.DS12.value] = [
    {"low": 0.05, "high": 0.21},
    {"low": 0.01, "high": 0.15},
]


def ga_mbkmeans_cluster_optimize(
    sheet_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame
):

    # Define the fitness function
    def fitness_fn(ga_instance, solution, solution_idx):
        try:
            pipeline = Pipeline(
                steps=[
                    ("feature_scaler", FeatureScaler(decimal_places=4)),
                    ("predictor", IT2TskPredictor(target=target_column)),
                ]
            )

            # Fit pipeline with the solution parameters
            pipeline.fit(
                train_df,
                predictor__clustering_method=clustering_method,
                predictor__mfs__cluster__batch_size=int(solution[0]),
                predictor__mfs__cluster__max_iter=int(solution[1]),
                predictor__mfs__cluster__tol=solution[2],
                predictor__mfs__cluster__max_no_improvement=int(
                    solution[3]
                ),  # Fixed for simplicity
                predictor__rules__cluster__batch_size=int(solution[0]),
                predictor__rules__cluster__max_iter=int(solution[1]),
                predictor__rules__cluster__tol=solution[2],
                predictor__rules__cluster__max_no_improvement=int(
                    solution[3]
                ),  # Fixed for simplicity
                predictor__mf_type=MFType2.GAUSSIAN,
                predictor__mf__builder__uncertainty_factor=solution[4],
                predictor__mf__builder__min_std_ratio=solution[5],
                predictor__linear_model=LinearModel.LSE,
            )

            # Transform test data and make predictions
            transformed_test_df = pipeline.transform(test_df)
            X_test = transformed_test_df.drop(columns=[target_column])
            y_test = transformed_test_df[target_column].values

            y_pred = pipeline.predict(X_test)

            # Calculate accuracy metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Get clustering times
            predictor = pipeline.named_steps["predictor"]
            rule_base = predictor.rule_base_
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

            # Multi-objective fitness: maximize R2 and minimize clustering time
            # Normalize clustering time (assuming max reasonable time is 60 seconds)
            normalized_time = min(total_clustering_time / 60.0, 1.0)

            # Multiplicative fitness: prioritize both accuracy, error, and speed
            fitness = (
                max(r2, 0) * (1 / (1 + mse)) * (1 / (1 + mae)) * (1 - normalized_time)
            )

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
                    "batch_size": {int(solution[0])},
                    "max_iter": {int(solution[1])},
                    "tol": {solution[2]},
                    "max_no_improvement": {int(solution[3])},
                    "uncertainty_factor": {solution[4]},
                    "min_std_ratio": {solution[5]},
                }
            )

            logging.info(
                f"Sheet: {sheet_name}  R2: {r2:.4f}, MSE: {mse:.4f}, Clustering Time: {total_clustering_time:.2f}s, Fitness: {fitness:.4f}"
            )

            return fitness

        except Exception as e:
            print(f"  Error in fitness evaluation: {str(e)}")
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

    n_samples = train_df.shape[0]

    # More focused parameter ranges
    gene_space = [
        # Batch size: from 5% to 25% of dataset size
        [max(256, n_samples // 10), max(512, n_samples // 8)],
        [100, 300, 500],  # max_iter
        {"low": 1e-6, "high": 1e-2, "step": 1e-6},  # tol
        [5, 10, 20],  # max_no_improvement
    ] + gene_spaces[sheet_name]

    num_genes = len(gene_space)
    sol_per_pop = 6  # Reduced population size
    num_parents_mating = sol_per_pop // 2
    num_generations = 25  # Reduced for faster execution

    parent_selection_type = "tournament"
    keep_parents = 1  # Keep the best parent at least one
    crossover_type = "uniform"
    mutation_type = "adaptive"
    mutation_num_genes = [3, 1]

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
            ("feature_scaler", FeatureScaler(decimal_places=4)),
            ("predictor", IT2TskPredictor(target=target_column)),
        ]
    )

    pipeline_best.fit(
        train_df,
        predictor__clustering_method=clustering_method,
        predictor__mfs__cluster__batch_size=int(solution[0]),
        predictor__mfs__cluster__max_iter=int(solution[1]),
        predictor__mfs__cluster__tol=solution[2],
        predictor__mfs__cluster__max_no_improvement=int(
            solution[3]
        ),  # Fixed for simplicity
        predictor__rules__cluster__batch_size=int(solution[0]),
        predictor__rules__cluster__max_iter=int(solution[1]),
        predictor__rules__cluster__tol=solution[2],
        predictor__rules__cluster__max_no_improvement=int(
            solution[3]
        ),  # Fixed for simplicity
        predictor__mf_type=MFType2.GAUSSIAN,
        predictor__mf__builder__uncertainty_factor=solution[4],
        predictor__mf__builder__min_std_ratio=solution[5],
        predictor__linear_model=LinearModel.LSE,
    )

    transformed_test_df_best = pipeline_best.transform(test_df)
    X_test_best = transformed_test_df_best.drop(columns=[target_column])
    y_test_best = transformed_test_df_best[target_column].values
    y_pred_best = pipeline_best.predict(X_test_best)

    r2_best = r2_score(y_test_best, y_pred_best)
    mse_best = mean_squared_error(y_test_best, y_pred_best)
    mae_best = mean_absolute_error(y_test_best, y_pred_best)

    predictor = pipeline_best.named_steps["predictor"]
    rule_base = predictor.rule_base_
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

    normalized_time = min(total_clustering_time / 60.0, 1.0)

    # Multiplicative fitness: prioritize both accuracy, error, and speed
    fitness = (
        max(r2_best, 0)
        * (1 / (1 + mse_best))
        * (1 / (1 + mae_best))
        * (1 - normalized_time)
    )

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
            "batch_size": {int(solution[0])},
            "max_iter": {int(solution[1])},
            "tol": {solution[2]},
            "max_no_improvement": {int(solution[3])},
            "uncertainty_factor": {solution[4]},
            "min_std_ratio": {solution[5]},
        }
    )

    logging.info("\n" + "=" * 60)
    logging.info("OPTIMIZATION COMPLETE")


start = time.time()
for i in range(10):
    logging.info(f"--- Iteration {i+1}/10 ---")
    for worksheet in WorkSheet:
        logging.info(f"\nProcessing worksheet: {worksheet.value}")
        dataset = Dataset(path=dataset_path, sheet_name=worksheet.value)
        train_df = dataset.train_df
        test_df = dataset.validate_df

        ga_mbkmeans_cluster_optimize(
            sheet_name=worksheet.value, train_df=train_df, test_df=test_df
        )

end = time.time()
elapsed = end - start

logging.info(
    f"Running tsk selection test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    f"Parameter_Tuning_10x_run_{clustering_method.value}.xlsx",
    "Parameter_Tuning",
)

logging.info("Script finished")
