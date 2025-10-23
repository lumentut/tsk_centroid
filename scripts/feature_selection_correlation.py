import logging
import sys
import time
import pandas as pd
from datetime import timedelta
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.dataset import Dataset, WorkSheet
from src.pipeline import Pipeline
from src.pipelines.transformers import CorrelationScoreSelector, FeatureScaler
from src.pipelines.predictors import IT2TskPredictor
from src.fis.fuzzy_logic.mfs import MFType2
from src.fis.fuzzy_logic.consequents import LinearModel
from src.clusters import ClusteringMethod
from src.utils.excel_operation import export_to_excel
from src.utils.correlation_score import correlation_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info("========================================")
logging.info("Feature Selection (Correlation Scores-based) Comparison Test")
logging.info("========================================\n")
logging.info("Script started")

dataset_path = "notebooks/data/e-nose_dataset_12_beef_cuts.xlsx"


target_column = "TVC"

statistics = []


def feature_selection(sheet_name: str, iteration_num: int):
    dataset = Dataset(path=dataset_path, sheet_name=sheet_name)

    train_df = dataset.train_df
    test_df = dataset.test_df

    correlation_scores, _ = correlation_score(train_df, target_column)
    data = [item for item in correlation_scores[:-2]]

    logging.info(f"Processing worksheet: {sheet_name}")

    def predict(feature_name: str, threshold: float):
        logging.info(
            f"Processing iteration {iteration_num} feature {feature_name} threshold {threshold} ..."
        )

        pipeline = Pipeline(
            steps=[
                ("feature_scaler", FeatureScaler(decimal_places=4)),
                ("feature_selection", CorrelationScoreSelector(target=target_column)),
                ("predictor", IT2TskPredictor(target=target_column)),
            ]
        )
        pipeline.fit(
            train_df,  # pipeline fit only for training dataframe
            feature_selection__threshold=threshold,
            predictor__clustering_method=ClusteringMethod.FUZZY_C_MEANS,
            predictor__mf_type=MFType2.GAUSSIAN,
            predictor__linear_model=LinearModel.LSE,
        )

        transformed_test_df = pipeline.transform(test_df)
        X_test_df = transformed_test_df.drop(columns=[target_column])
        y_test = transformed_test_df[target_column].values

        y_pred = pipeline.predict(X_test_df)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        predictor = pipeline.named_steps["predictor"]
        rule_base = predictor.rule_base_
        io_vars = predictor.io_vars_

        selected_features = pipeline.named_steps["feature_selection"].selected_features_

        statistics.append(
            {
                "Worksheet Name": sheet_name,
                "Threshold": threshold,
                "Selected Features": len(selected_features),
                "R2": r2,
                "MSE": mse,
                "MAE": mae,
                "Selected Feature Names": ", ".join(selected_features),
                "total_rule_base_clustering_time": rule_base.get_execution_time_stats(
                    "_clusterize_data"
                )["total_time"],
                "total_rules": len(rule_base.rules_),
                "total_features_clustering_time": io_vars.get_execution_time_stats(
                    "_clusterize_data"
                )["total_time"],
                "average_feature_clustering_time": io_vars.get_execution_time_stats(
                    "_clusterize_data"
                )["average_time"],
                "total_features": len(io_vars.input_variables_)
                + len(io_vars.output_variables_),
                "Total Training Samples": len(train_df),
                "Total Testing Samples": len(test_df),
                "Total Samples": len(train_df) + len(test_df),
                "total pipeline fit time": pipeline.get_execution_time_stats("fit")[
                    "total_time"
                ],
                "total pipeline transform time": pipeline.get_execution_time_stats(
                    "transform"
                )["total_time"],
                "total pipeline predict time": pipeline.get_execution_time_stats(
                    "predict"
                )["total_time"],
            }
        )

    for _, item in enumerate(data):
        feature_name = list(item.keys())[0]
        threshold = list(item.values())[0]
        predict(feature_name=feature_name, threshold=threshold)


start = time.time()
for worksheet in WorkSheet:
    for i in range(1000):
        feature_selection(sheet_name=worksheet.value, iteration_num=i)

end = time.time()
elapsed = end - start

logging.info(
    f"Running feature selection correlation test... Elapsed time: {timedelta(seconds=elapsed)}"
)

logging.info("Done.")

statistics_df = pd.DataFrame(statistics)

export_to_excel(
    statistics_df,
    "results_e-nose_dataset_12_beef_cuts.xlsx",
    "feature_selection_correlation",
)

logging.info("Script finished")
