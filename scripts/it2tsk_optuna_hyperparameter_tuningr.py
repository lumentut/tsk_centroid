import logging
import sys
import time
import pandas as pd
import numpy as np
import optuna
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
fcm_config_file_path = "notebooks/experiments/TSK_FCM_Exploration.xlsx"
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
    return {
        item["Sheet Name"]: {
            k: item[k] for k in ["batch_size", "tol", "max_no_improvement"]
        }
        for item in selected_max_df.to_dict(orient="records")
    }


def get_fcm_params() -> dict:
    df = pd.read_excel(fcm_config_file_path)
    df["Sheet_Order"] = df["Sheet Name"].str.extract(r"^(\d+)").astype(int)
    df_max = df.loc[df.groupby("Sheet Name")["R2"].idxmax()]
    df_max[["R2"]] = df_max[["R2"]].round(4)

    df_sorted = df_max.sort_values("Sheet_Order").reset_index(drop=True)
    selected_max_df = df_sorted[["Sheet Name", "R2", "m", "error", "max_iter"]]

    selected_max_df.to_dict(orient="records")
    result = {
        item["Sheet Name"]: {k: item[k] for k in ["m", "error", "max_iter"]}
        for item in selected_max_df.to_dict(orient="records")
    }
    return result


def get_gaussian_params() -> dict:
    df = pd.read_excel(gaussian_config_file_path, sheet_name="uf_msr_summary")
    return {
        item["Sheet Name"]: {
            k: item[k] for k in ["UF_min", "UF_max", "MSR_min", "MSR_max", "R2_max"]
        }
        for item in df.to_dict(orient="records")
    }


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info("=" * 80)
logging.info("IT2TSK Gaussian MF exploration (Optuna-based optimization)")
logging.info("=" * 80)

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"
target_column = "TVC"
clustering_method = ClusteringMethod.FUZZY_C_MEANS

statistics = []


def objective(
    trial,
    sheet_name,
    gaussian_params,
    # mbk_params,
    fcm_params,
    transformer_pipe,
    transformed_train_df,
    test_df,
    clusters,
):
    try:
        # Sample parameters
        uncertainty_factor = trial.suggest_float(
            "uncertainty_factor",
            gaussian_params.get("UF_min"),
            gaussian_params.get("UF_max"),
            step=0.01,
        )
        min_std_ratio = trial.suggest_float(
            "min_std_ratio",
            gaussian_params.get("MSR_min"),
            gaussian_params.get("MSR_max"),
            step=0.01,
        )

        # Build pipeline
        pipeline = Pipeline(
            steps=[
                ("feature_scaler", FeatureScaler(decimal_places=4)),
                ("predictor", IT2TskPredictor(target=target_column)),
            ]
        )

        pipeline.fit(
            transformed_train_df,
            predictor__clusters=clusters,
            predictor__mf_type=MFType2.GAUSSIAN,
            predictor__mf__builder__uncertainty_factor=uncertainty_factor,
            predictor__mf__builder__min_std_ratio=min_std_ratio,
            predictor__linear_model=LinearModel.LSE,
        )

        # Evaluate
        transformed_test_df = transformer_pipe.transform(test_df)
        X_test = transformed_test_df.drop(columns=[target_column])
        y_test = transformed_test_df[target_column].values
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        predictor = pipeline.named_steps["predictor"]
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
        total_features = len(predictor.io_vars_.input_variables_) + len(
            predictor.io_vars_.output_variables_
        )

        normalized_time = min(total_clustering_time / 60.0, 1.0)

        # Fitness = maximize R² * minimize time/error
        fitness = max(r2, 0) * (1 / (1 + mse)) * (1 / (1 + mae)) * (1 - normalized_time)

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
                "m": float(fcm_params.get("m")),
                "error": float(fcm_params.get("error")),
                "max_iter": int(fcm_params.get("max_iter")),
                # "batch_size": int(mbk_params.get("batch_size")),
                # "tol": mbk_params.get("tol"),
                # "max_no_improvement": int(mbk_params.get("max_no_improvement")),
                "uncertainty_factor": uncertainty_factor,
                "min_std_ratio": min_std_ratio,
            }
        )

        logging.info(
            f"[Trial {trial.number}] Sheet={sheet_name} | R2={r2:.4f} | Fitness={fitness:.4f} | Time={total_clustering_time:.2f}s"
        )

        r2_threshold = gaussian_params.get("R2_max", 0.0)
        if r2 < r2_threshold:
            trial.set_user_attr("below_threshold", True)
            return -1000 + r2 * 100

        return fitness

    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {str(e)}")
        return -1000


start = time.time()

for worksheet in WorkSheet:
    logging.info(f"=== Optimizing for Sheet: {worksheet.value} ===")

    dataset = Dataset(path=dataset_path, sheet_name=worksheet.value)
    train_df = dataset.train_df
    test_df = dataset.validate_df

    # mbk_params = get_mbkmeans_params().get(worksheet.value, {})
    fcm_params = get_fcm_params().get(worksheet.value, {})
    gaussian_params = get_gaussian_params().get(worksheet.value, {})

    transformer_pipe = Pipeline(
        steps=[
            ("feature_scaler", FeatureScaler(decimal_places=4)),
            (
                "clusterer",
                Clusterer(
                    method=clustering_method,
                    # batch_size=int(mbk_params.get("batch_size")),
                    # tol=mbk_params.get("tol"),
                    # max_no_improvement=int(mbk_params.get("max_no_improvement")),
                    m=float(fcm_params.get("m")),
                    error=float(fcm_params.get("error")),
                    max_iter=int(fcm_params.get("max_iter")),
                ),
            ),
        ]
    )

    transformer_pipe.fit(train_df)
    transformed_train_df = transformer_pipe.transform(train_df)
    clusters = transformer_pipe.named_steps["clusterer"].clusters

    for run_idx in range(10):
        logging.info(f"=== Run {run_idx+1}/10 for Sheet: {worksheet.value} ===")
        # --- Optuna Study ---
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            lambda trial: objective(
                trial,
                worksheet.value,
                gaussian_params,
                # mbk_params,
                fcm_params,
                transformer_pipe,
                transformed_train_df,
                test_df,
                clusters,
            ),
            n_trials=30,  # Number of Optuna trials (instead of generations)
            timeout=3600,  # 1-hour timeout safeguard
            show_progress_bar=True,
        )

        best_trial = study.best_trial
        logging.info(f"Best trial for {worksheet.value}: {best_trial.params}")
        logging.info(f"Best fitness: {best_trial.value:.4f}")

end = time.time()
elapsed = end - start
logging.info(f"Total elapsed time: {timedelta(seconds=elapsed)}")

statistics_df = pd.DataFrame(statistics)
export_to_excel(
    statistics_df, "IT2TSK_Optuna_Hyperparameter_Tuning.xlsx", "performances"
)

logging.info("Optimization complete and results exported.")
