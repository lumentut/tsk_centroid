import inspect
import pandas as pd
from src.pipelines import BaseTransformer
from src.utils.timing import Timing, execution_time


class Pipeline(Timing):
    """
    A Pipeline class that mimics scikit-learn's Pipeline functionality.

    Pipeline of transformers with a final estimator.
    Sequentially applies a list of transformers and a final estimator.
    Intermediate steps of the pipeline must be 'transformers', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    """

    def __init__(self, steps):
        """
        Initialize the Pipeline.

        Args:
            steps (list of tuple): List of (name, transform) tuples (implementing fit/transform) that are
            chained, in the order in which they are chained, with the last object
            an estimator.
        """
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        """Validate that all steps have the required methods."""
        if not self.steps:
            raise ValueError("Pipeline cannot be empty")

        # Check that all but the last step have transform method
        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"All intermediate steps should implement fit and transform. "
                    f"'{name}' (type {type(step)}) doesn't."
                )

        # Check that the last step has fit method
        name, step = self.steps[-1]
        if not hasattr(step, "fit"):
            raise TypeError(
                f"Last step should implement fit. "
                f"'{name}' (type {type(step)}) doesn't."
            )

    @property
    def named_steps(self):
        """Access the steps by name."""
        return dict(self.steps)

    def __getitem__(self, index):
        """Get a step by index or name."""
        if isinstance(index, str):
            return self.named_steps[index]
        return self.steps[index][1]

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self.steps)

    @execution_time
    def fit(self, df: pd.DataFrame, **fit_params) -> "Pipeline":
        """
        Fit the model.

        Fit all the transforms one after the other and transform the train data,
        then fit the transformed train data using the final estimator to build the model.

        Args:
            df (pd.DataFrame): The input training data to fit.

        Returns:
            self (Pipeline): This Pipeline instance.
        """
        transformed_df = df.copy()

        self.learned_params = {}

        # Fit and transform all steps except the last one
        for _, (name, transformer) in enumerate(self.steps[:-1]):
            step_fit_params = self._get_fit_params_for_step(name, fit_params)

            # Validate the fit parameters
            self._validate_step_fit_params(transformer, step_fit_params)

            # Fit the transformer
            transformer.fit(transformed_df, **step_fit_params)

            self.learned_params[name] = self._get_learned_params(transformer)

            # Transform the train data for the next step
            transformed_df = transformer.transform(transformed_df)

        # Fit the final estimator
        final_name, final_estimator = self.steps[-1]
        final_fit_params = self._get_fit_params_for_step(final_name, fit_params)
        final_estimator.fit(transformed_df, **final_fit_params)

        return self

    @execution_time
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the test data.

        Apply transforms to the data. Skip steps that don't have transform method
        (typically the final estimator).
        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        transformed_df = df.copy()
        # Transform all steps that have transform method exclude final step
        for _, transformer in self.steps:
            if hasattr(transformer, "transform"):
                transformed_df = transformer.transform(transformed_df)

        return transformed_df

    @execution_time
    def predict(self, input_df: pd.DataFrame):
        """Make predictions using the pipeline.

        Args:
            df (pd.DataFrame): DataFrame containing the input features for prediction.

        Returns:
            pd.Series: Predicted values.
        """
        # Apply all transformations (this is what pipeline.transform() does)
        # transformed_df = self.transform(df)

        # Predict using final estimator
        final_name, final_estimator = self.steps[-1]
        if not hasattr(final_estimator, "predict"):
            raise ValueError(
                f"Final step '{final_name}' does not have a predict method"
            )

        return final_estimator.predict(input_df)

    def _get_fit_params_for_step(
        self, step_name: str, fit_params: dict
    ) -> dict[str, any]:
        """Extract fit parameters for a specific step."""
        step_fit_params = {}
        for param_name, param_value in fit_params.items():
            if param_name.startswith(step_name + "__"):
                # Remove the step name prefix
                clean_param_name = param_name[len(step_name) + 2 :]
                step_fit_params[clean_param_name] = param_value
        return step_fit_params

    def _validate_step_fit_params(
        self, transformer: BaseTransformer, step_fit_params: dict
    ) -> None:
        """Inspect fit method to validate parameters."""
        fit_signature = inspect.signature(transformer.fit)
        expected_params = [
            p for p in fit_signature.parameters.keys() if p not in ["self", "df"]
        ]

        for param_name in step_fit_params.keys():
            if param_name not in expected_params:
                # Check if method accepts **kwargs
                has_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in fit_signature.parameters.values()
                )

                if not has_kwargs:
                    raise ValueError(f"Parameter '{param_name}' not accepted")

    def _get_learned_params(self, transformer: BaseTransformer) -> dict[str, any]:
        """Get the learned parameters from the fitted transformer."""
        learned_params = {}

        for attr_name in dir(transformer):
            if attr_name.endswith("_") and not attr_name.startswith("_"):
                attr_value = getattr(transformer, attr_name)
                # Skip methods
                if not callable(attr_value):
                    learned_params[attr_name] = attr_value

        return learned_params

    def __repr__(self):
        """Return a string representation of the Pipeline."""
        steps_str = ",\n ".join(
            [f"('{name}', {repr(step)})" for name, step in self.steps]
        )
        return f"Pipeline(steps=[{steps_str}])"
