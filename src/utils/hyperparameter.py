import pandas as pd

notebook_tuned_params_file_path = "experiments/IT2TSK_Hyperparameter_Tuning.xlsx"
tuned_params_file_path = "notebooks/experiments/IT2TSK_Hyperparameter_Tuning.xlsx"


def get_tuned_params() -> dict:
    try:
        df = pd.read_excel(notebook_tuned_params_file_path)
    except FileNotFoundError:
        df = pd.read_excel(tuned_params_file_path)

    df["Sheet_Order"] = df["Sheet Name"].str.extract(r"^(\d+)").astype(int)
    df_max = df.loc[df.groupby("Sheet Name")["R2"].idxmax()]
    df_max[["R2"]] = df_max[["R2"]].round(4)

    df_sorted = df_max.sort_values("Sheet_Order").reset_index(drop=True)
    selected_max_df = df_sorted[
        [
            "Sheet Name",
            "R2",
            "batch_size",
            "tol",
            "max_no_improvement",
            "uncertainty_factor",
            "min_std_ratio",
        ]
    ]

    selected_max_df.to_dict(orient="records")
    result = {
        item["Sheet Name"]: {
            k: item[k]
            for k in [
                "batch_size",
                "tol",
                "max_no_improvement",
                "uncertainty_factor",
                "min_std_ratio",
            ]
        }
        for item in selected_max_df.to_dict(orient="records")
    }
    return result
