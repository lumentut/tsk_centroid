def get_sheet_names(excel_path):
    """Returns the sheet names of the given Excel file."""
    import pandas as pd

    try:
        xls = pd.ExcelFile(excel_path)
        return xls.sheet_names
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        return []


def train_test_df(
    df, label_col="Label", stratify_col=None, test_size=0.2, random_state=42
):
    """Splits the dataframe into training and testing sets."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col else None,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
