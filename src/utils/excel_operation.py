import pandas as pd


def export_to_excel(df: pd.DataFrame, filename: str, sheet_name: str):
    try:
        with pd.ExcelWriter(
            filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
