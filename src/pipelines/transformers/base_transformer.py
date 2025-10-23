from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.
    
    This class defines the interface that all transformer classes should implement.
    Transformers typically follow a fit-transform pattern where:
    1. fit() learns parameters from the data
    2. transform() applies the transformation using learned parameters
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.is_fitted_ = False # not implemented
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, **fit_args: Any) -> 'BaseTransformer':
        """
        Fit the transformer to the data.
        
        This method should learn any necessary parameters from the input data
        and store them as instance attributes.
        
        Args:
            df (pd.DataFrame): The input DataFrame to fit the transformer on.
            **fit_args (Any): Additional keyword arguments for fitting.

        Returns:
            BaseTransformer: Returns self to allow method chaining.
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.
        
        This method applies the transformation using parameters learned during fit().
        The transformer must be fitted before calling this method.

        Args:
            df (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
            
        Raises:
            NotFittedError: If the transformer has not been fitted yet.
        """
        pass
   
    def _get_numeric_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of numeric columns from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of numeric column names
        """
        return [col for col in df.select_dtypes(include='number').columns]

    def fit_transform(self, df: pd.DataFrame, **fit_args: Any) -> pd.DataFrame:
        """
        Fit the transformer and transform the data in one step.
        
        This is a convenience method that calls fit() followed by transform().

        Args:
            df (pd.DataFrame): The input DataFrame to fit and transform.
            **fit_args (Any): Additional keyword arguments for fitting.

        Returns:
            pd.DataFrame: The transformed DataFrame.
            The transformed DataFrame.
        """
        return self.fit(df, **fit_args).transform(df)
    
    def get_params(self):
        """
        Get parameters for this transformer.

        Returns:
            dict: Parameter names mapped to their values.
        """
        # Get all public attributes (parameters) of the transformer
        params = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)
                # Skip fitted attributes and internal state
                if not key.endswith('_'):
                    params[key] = value
        return params
    
    def set_params(self, **params):
        """
        Set parameters for this transformer.

        Args:
            **params (dict): Parameters to set.

        Returns:
            BaseTransformer: This transformer instance.
        """
        for param_name, param_value in params.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                raise ValueError(f"Invalid parameter {param_name} for transformer {type(self).__name__}")
        return self
    
    def _check_is_fitted(self) -> None:
        """
        Check if the transformer has been fitted.
        
        Raises:
            NotFittedError: If the transformer has not been fitted yet.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this transformer."
            )

class NotFittedError(ValueError):
    """
    Exception raised when a transformer is used before being fitted.
    
    This exception is raised when transform() or other methods that require
    the transformer to be fitted are called before fit() has been called.
    """
    pass