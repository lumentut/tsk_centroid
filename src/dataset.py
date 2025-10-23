import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split


class WorkSheet(Enum):
    DS1 = "1.Inside-Outside"
    DS2 = "2.Round"
    DS3 = "3.Top_Sirloin"
    DS4 = "4.Tenderloin"
    DS5 = "5.Flap_meat"
    DS6 = "6.Striploin"
    DS7 = "7.Rib_eye"
    DS8 = "8.Skirt_meat"
    DS9 = "9.Brisket"
    DS10 = "10.Clod_Chuck"
    DS11 = "11.Shin"
    DS12 = "12.Fat"


class Dataset:
    def __init__(
        self,
        path: str = "data/Beef_TVC_Dataset.xlsx",
        sheet_name: str = WorkSheet.DS1.value,
        stratiify_col: str = "Label",
        sort_column: str = "Minute",
        skip_columns: list = ["Minute", "Label"],
        train_size: float = 0.7,
        random_state: int = 42,
    ):
        """Dataset class for loading and handling datasets.

        Args:
            path (str): Path to the dataset file.
            sheet_name (str): Sheet name for Excel files. Defaults to WorkSheet.DS1.value.
            stratiify_col (str): Column to use for stratified sampling. Defaults to "Label".
            sort_column (str): Column to sort the datasets by. Defaults to "Minute".
            skip_columns (list): Columns to exclude from final datasets. Defaults to ["Minute", "Label"].
            train_size (float): Proportion of data to use for training. Defaults to 0.8.
                                The remaining data (1-train_size) is split equally between validation and test sets.
            random_state (int): Random seed for reproducibility. Defaults to 42.
        """
        # Load the dataset from Excel
        self.df = pd.read_excel(path, sheet_name=sheet_name)

        # Store configuration parameters
        self.stratify_col = stratiify_col
        self.sort_column = sort_column
        self.skip_columns = skip_columns
        self.train_size = train_size
        self.random_state = random_state

        # Split the dataset into training, validation, and testing sets
        self._train_df, self._validate_df, self._test_df = (
            self.create_train_test_split()
        )

    @property
    def train_df(self):
        """Get a copy of the training dataset."""
        return self._train_df.copy()

    @property
    def validate_df(self):
        """Get a copy of the validation dataset."""
        return self._validate_df.copy()

    # Add an alias for validation dataset - more standard naming
    @property
    def val_df(self):
        """Get a copy of the validation dataset (alias for validate_df)."""
        return self._validate_df.copy()

    @property
    def test_df(self):
        """Get a copy of the testing dataset."""
        return self._test_df.copy()

    def create_train_test_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, and test splits of the dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing the training, validation, and testing DataFrames.
        """
        sheet_df = self.df.copy()

        # First split: separate training data (80%) from the rest (20%)
        train_df, val_test_df = self._split_stratified(
            sheet_df, train_size=self.train_size
        )

        # Second split: divide the remaining 20% equally between validation and test (50/50 split)
        # This makes validation 10% and testing 10% of the original dataset
        val_df, test_df = self._split_stratified(val_test_df, train_size=0.5)

        # Remove unnecessary columns
        train_df = self._drop_and_reset(train_df)
        val_df = self._drop_and_reset(val_df)
        test_df = self._drop_and_reset(test_df)

        return train_df, val_df, test_df

    def _split_stratified(
        self, df: pd.DataFrame, train_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Helper method to perform a stratified split on a dataframe.

        Args:
            df: DataFrame to split
            train_size: Proportion to use for the training split

        Returns:
            Tuple of (training_split, testing_split) DataFrames
        """
        X = df.drop(columns=[self.stratify_col], axis=1)
        y = df[self.stratify_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=self.random_state, stratify=y
        )

        train_df = (
            pd.concat([X_train, y_train], axis=1)
            .sort_values(by=self.sort_column, ascending=True)
            .reset_index(drop=True)
        )

        test_df = (
            pd.concat([X_test, y_test], axis=1)
            .sort_values(by=self.sort_column, ascending=True)
            .reset_index(drop=True)
        )

        return train_df, test_df

    def _drop_and_reset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove skip columns and reset index.

        Args:
            df: DataFrame to process

        Returns:
            Processed DataFrame
        """
        return df.drop(columns=self.skip_columns, axis=1)
