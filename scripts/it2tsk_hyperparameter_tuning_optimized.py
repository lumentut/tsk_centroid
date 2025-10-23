import os
import sys
import time
import logging
import multiprocessing as mp
from datetime import timedelta

import numpy as np
import pandas as pd
import pygad
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

# --------------------------- CONFIG & LOGGING ---------------------------

mbk_config_file_path = "notebooks/experiments/IT2TSK_MBKMeans_Exploration.xlsx"
gaussian_config_file_path = "notebooks/experiments/IT2TSK_Gaussian_Exploration.xlsx"
dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"
target_column = "TVC"
clustering_method = ClusteringMethod.MBKMEANS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Prefer fork on POSIX so workers inherit globals (faster & pickling-free)
if os.name == "posix":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # already set

# --------------------------- PARAM LOADERS ---------------------------


def get_mbkmeans_params() -> dict:
    df = pd.read_excel(mbk_config_file_path)
    df["Sheet_Order"] = df["Sheet Name"].str.extract(r"^(\d+)").astype(int)
    df_max = df.loc[df.groupby("Sheet Name")["R2"].idxmax()]
    df_max[["R2"]] = df_max[["R2"]].round(4)
    df_sorted = df_max.sort_values("Sheet_Order").reset_index(drop=True)
    selected_max_df = df_sorted[
        ["Sheet Name", "R2", "batch_size", "tol", "max_no_improvement"]
    ]
    return {
        item["Sheet Name"]: {
            k: item[k] for k in ["batch_size", "tol", "max_no_improvement"]
        }
        for item in selected_max_df.to_dict(orient="records")
    }


def get_gaussian_params() -> dict:
    df = pd.read_excel(gaussian_config_file_path, sheet_name="uf_msr_summary")
    return {
        item["Sheet Name"]: {
            k: item[k] for k in ["UF_min", "UF_max", "MSR_min", "MSR_max", "R2_max"]
        }
        for item in df.to_dict(orient="records")
    }


# --------------------------- FITNESS CONTEXT ---------------------------
# We use a module-level context so pygad workers can access it (especially with fork).

_CTX = {
    "sheet_name": None,
    "clusters": None,  # Clusters
    "transformed_train_df": None,  # pd.DataFrame
    "X_test": None,  # pd.DataFrame or np.ndarray
    "y_test": None,  # np.ndarray
    "mbk_params": None,  # dict
    "gaussian_params": None,  # dict for bounds + R2 threshold
    "transformer_pipe": None,  # Pipeline (for consistency; test already transformed)
    "memo": None,  # dict for memoization
}


def setup_fitness_context(
    sheet_name: str,
    clusters: Clusters,
    transformed_train_df: pd.DataFrame,
    X_test,
    y_test,
    mbk_params: dict,
    gaussian_params: dict,
    transformer_pipe: Pipeline,
):
    # Called per sheet/iteration before GA.run()
    _CTX["sheet_name"] = sheet_name
    _CTX["clusters"] = clusters
    _CTX["transformed_train_df"] = transformed_train_df
    _CTX["X_test"] = X_test
    _CTX["y_test"] = y_test
    _CTX["mbk_params"] = mbk_params
    _CTX["gaussian_params"] = gaussian_params
    _CTX["transformer_pipe"] = transformer_pipe
    _CTX["memo"] = {}  # clear memo every run


# --------------------------- FITNESS FUNCTION ---------------------------


def it2tsk_fitness(ga_instance, solution, solution_idx):
    """Top-level (picklable) fitness for pygad parallel workers."""
    try:
        uf = float(solution[0])
        msr = float(solution[1])

        # Build predictor and fit on already-transformed train set
        predictor_pipe = Pipeline(
            steps=[("predictor", IT2TskPredictor(target=target_column))]
        )

        predictor_pipe.fit(
            _CTX["transformed_train_df"],
            predictor__clusters=_CTX["clusters"],
            predictor__mf_type=MFType2.GAUSSIAN,
            predictor__mf__builder__uncertainty_factor=uf,
            predictor__mf__builder__min_std_ratio=msr,
            predictor__linear_model=LinearModel.LSE,
        )

        # Predict on precomputed test arrays
        y_pred = predictor_pipe.predict(_CTX["X_test"])

        y_test = _CTX["y_test"]
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Clustering time stats (clusters are reused; stats live in predictor.clusters_)
        predictor = predictor_pipe.named_steps["predictor"]
        total_rule_base_clustering_time = predictor.clusters_.get_execution_time_stats(
            "_create_rules_cluster"
        )["total_time"]
        total_features_clustering_time = predictor.clusters_.get_execution_time_stats(
            "_create_mfs_clusters"
        )["total_time"]
        total_clustering_time = (
            total_features_clustering_time + total_rule_base_clustering_time
        )

        # Normalize clustering time to [0,1] with a 60s cap
        normalized_time = min(total_clustering_time / 60.0, 1.0)

        # Multi-objective, multiplicative fitness
        fitness = (
            max(r2, 0.0)
            * (1.0 / (1.0 + mse))
            * (1.0 / (1.0 + mae))
            * (1.0 - normalized_time)
        )

        return fitness

    except Exception as e:
        # Return very low fitness on failure
        # Avoid noisy logs inside parallel workers
        return -1000.0


# --------------------------- GA RUN PER SHEET ---------------------------


def run_ga_for_sheet(
    sheet_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mbk_params: dict,
    gaussian_params: dict,
    n_runs: int = 10,
    num_generations: int = 25,
    sol_per_pop: int = 6,
    max_workers: int | None = None,
):
    """
    Runs GA multiple times for a sheet and returns a list of best results dicts.
    Optimizations:
      * Precompute transform(X_test, y_test) once per sheet.
      * Reuse clusters built once per sheet.
      * Memoization inside each GA run.
      * Proper parallel_processing with top-level fitness.
    """
    # Build transformer+clusterer ONCE per sheet
    transformer_pipe = Pipeline(
        steps=[
            ("feature_scaler", FeatureScaler(decimal_places=4)),
            (
                "clusterer",
                Clusterer(
                    method=clustering_method,
                    batch_size=int(mbk_params.get("batch_size")),
                    tol=mbk_params.get("tol"),
                    max_no_improvement=int(mbk_params.get("max_no_improvement")),
                ),
            ),
        ]
    )
    transformer_pipe.fit(train_df)

    transformed_train_df = transformer_pipe.transform(train_df)
    clusters = transformer_pipe.named_steps["clusterer"].clusters

    # Precompute transformed test once
    transformed_test_df = transformer_pipe.transform(test_df)
    X_test = transformed_test_df.drop(columns=[target_column])
    y_test = transformed_test_df[target_column].values

    # GA search space per sheet from gaussian_params
    gene_space = [
        {
            "low": gaussian_params.get("UF_min"),
            "high": gaussian_params.get("UF_max"),
            "step": 0.01,
        },  # UF
        {
            "low": gaussian_params.get("MSR_min"),
            "high": gaussian_params.get("MSR_max"),
            "step": 0.01,
        },  # MSR
    ]
    num_genes = 2
    num_parents_mating = max(1, sol_per_pop // 2)

    # Parallel degree inside pygad; avoid oversubscription if caller also parallelizes sheets
    # If max_workers is provided by caller, keep pygad at that; else use all cores.
    parallel_workers = max_workers or mp.cpu_count()

    best_results = []

    for run_idx in range(n_runs):
        # New memo per run to avoid stale cache bias
        setup_fitness_context(
            sheet_name=sheet_name,
            clusters=clusters,
            transformed_train_df=transformed_train_df,
            X_test=X_test,
            y_test=y_test,
            mbk_params=mbk_params,
            gaussian_params=gaussian_params,
            transformer_pipe=transformer_pipe,
        )

        # Callback to capture tracking (lightweight)
        generation_history = []
        best_fitness_history = []
        best_params_history = []

        def on_generation(ga_instance):
            gen = ga_instance.generations_completed
            best_sol, best_fit, _ = ga_instance.best_solution()
            generation_history.append(gen)
            best_fitness_history.append(best_fit)
            best_params_history.append(tuple(map(float, best_sol)))

        ga = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=it2tsk_fitness,  # top-level, picklable
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            parent_selection_type="tournament",
            keep_parents=1,
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_num_genes=[2, 1],
            on_generation=on_generation,
            stop_criteria=["saturate_5"],
            parallel_processing=parallel_workers,  # let pygad parallelize fitness
        )

        ga.run()
        solution, solution_fitness, _ = ga.best_solution()

        # Evaluate best solution precisely (once) for detailed stats
        uf_best, msr_best = float(solution[0]), float(solution[1])

        predictor_pipe = Pipeline(
            steps=[("predictor", IT2TskPredictor(target=target_column))]
        )
        predictor_pipe.fit(
            transformed_train_df,
            predictor__clusters=clusters,
            predictor__mf_type=MFType2.GAUSSIAN,
            predictor__mf__builder__uncertainty_factor=uf_best,
            predictor__mf__builder__min_std_ratio=msr_best,
            predictor__linear_model=LinearModel.LSE,
        )
        y_pred_best = predictor_pipe.predict(X_test)

        r2_best = r2_score(y_test, y_pred_best)
        mse_best = mean_squared_error(y_test, y_pred_best)
        mae_best = mean_absolute_error(y_test, y_pred_best)

        predictor = predictor_pipe.named_steps["predictor"]
        total_rule_base_clustering_time = predictor.clusters_.get_execution_time_stats(
            "_create_rules_cluster"
        )["total_time"]
        total_features_clustering_time = predictor.clusters_.get_execution_time_stats(
            "_create_mfs_clusters"
        )["total_time"]
        total_clustering_time = (
            total_features_clustering_time + total_rule_base_clustering_time
        )
        avg_feature_clustering_time = predictor.clusters_.get_execution_time_stats(
            "_create_mfs_clusters"
        )["average_time"]
        total_features = len(predictor.io_vars_.input_variables_) + len(
            predictor.io_vars_.output_variables_
        )

        normalized_time = min(total_clustering_time / 60.0, 1.0)
        fitness_best = (
            max(r2_best, 0.0)
            * (1.0 / (1.0 + mse_best))
            * (1.0 / (1.0 + mae_best))
            * (1.0 - normalized_time)
        )

        best_results.append(
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
                "Average Feature Clustering Time": avg_feature_clustering_time,
                "Total Features": total_features,
                "Fitness": fitness_best,
                "batch_size": int(mbk_params.get("batch_size")),
                "tol": mbk_params.get("tol"),
                "max_no_improvement": int(mbk_params.get("max_no_improvement")),
                "uncertainty_factor": uf_best,
                "min_std_ratio": msr_best,
                "generation_history": generation_history,
                "best_fitness_history": best_fitness_history,
                "best_params_history": best_params_history,
            }
        )

        logging.info(
            f"[{sheet_name}] Run {run_idx+1}/{n_runs} -> "
            f"R2={r2_best:.4f} MSE={mse_best:.4f} MAE={mae_best:.4f} "
            f"R2={r2_best:.4f} "
            f"UF={uf_best:.2f} MSR={msr_best:.2f} Fit={fitness_best:.4f} "
            f"ClusterTime={total_clustering_time:.2f}s"
        )

    return best_results


# --------------------------- MAIN ---------------------------


def main():
    logging.info("=" * 73)
    logging.info("IT2TSK Gaussian MF exploration — Optimized")
    logging.info("=" * 73)
    logging.info("Script started")

    start = time.time()
    statistics = []

    # Load experiment configs ONCE
    mbk_params_all = get_mbkmeans_params()
    gaussian_params_all = get_gaussian_params()

    # You can tune these
    N_RUNS_PER_SHEET = 10
    NUM_GENERATIONS = 25
    SOL_PER_POP = 6
    # To avoid oversubscription when you later parallelize across sheets:
    INTERNAL_WORKERS = mp.cpu_count()

    for worksheet in WorkSheet:
        sheet_name = worksheet.value
        dataset = Dataset(path=dataset_path, sheet_name=sheet_name)
        train_df = dataset.train_df
        test_df = dataset.validate_df

        mbk_params = mbk_params_all.get(sheet_name, {})
        gaussian_params = gaussian_params_all.get(sheet_name, {})

        results = run_ga_for_sheet(
            sheet_name=sheet_name,
            train_df=train_df,
            test_df=test_df,
            mbk_params=mbk_params,
            gaussian_params=gaussian_params,
            n_runs=N_RUNS_PER_SHEET,
            num_generations=NUM_GENERATIONS,
            sol_per_pop=SOL_PER_POP,
            max_workers=INTERNAL_WORKERS,
        )
        statistics.extend(results)

    elapsed = time.time() - start
    logging.info(f"Elapsed time: {timedelta(seconds=elapsed)}")
    logging.info("Done.")

    statistics_df = pd.DataFrame(statistics)
    export_to_excel(statistics_df, "IT2TSK_Hyperparameter_Tuning.xlsx", "performances")
    logging.info("Script finished")


if __name__ == "__main__":
    main()
