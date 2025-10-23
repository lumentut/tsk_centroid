from typing import Union
import inspect

from src.fis.fuzzy_logic.mfs.type_1.t1_base_mf_builder import MFType1
from .type_2 import (
    T2GaussianMFBuilder,
    T2TriangularMFBuilder,
    T2TrapezoidalMFBuilder,
    MFType2,
)
from .type_1 import (
    T1GaussianMFBuilder,
    T1TriangularMFBuilder,
    T1TrapezoidalMFBuilder,
    MFType1,
)

MFBuilder = Union[
    T2GaussianMFBuilder,
    T2TriangularMFBuilder,
    T1GaussianMFBuilder,
    T1TriangularMFBuilder,
    T1TrapezoidalMFBuilder,
]

MFType = Union[MFType1, MFType2]


class MFFactory:
    """Factory class for creating membership function builders."""

    _mf_builders = {
        MFType1.GAUSSIAN: T1GaussianMFBuilder,
        MFType1.TRIANGULAR: T1TriangularMFBuilder,
        MFType1.TRAPEZOIDAL: T1TrapezoidalMFBuilder,
        MFType2.GAUSSIAN: T2GaussianMFBuilder,
        MFType2.TRIANGULAR: T2TriangularMFBuilder,
        MFType2.TRAPEZOIDAL: T2TrapezoidalMFBuilder,
    }

    _kwargs_prefix = "mf__builder__"

    @classmethod
    def create_mf_builder(cls, name: MFType, **kwargs) -> MFBuilder:
        """Create a new instance of a registered mf builder.

        Args:
            name (str): The name of the membership function builder.
            **kwargs: Additional parameters to pass to the builder's constructor.

        Returns:
            MfBuilder: A new instance of the requested mf builder.

        Raises:
            ValueError: If the requested mf builder is not registered.
        """
        builder_class = cls._mf_builders.get(name)
        if builder_class is not None:
            builder_kwargs = {}
            for k, v in kwargs.items():
                if k.startswith(cls._kwargs_prefix):
                    new_key = k[len(cls._kwargs_prefix) :]
                    builder_kwargs[new_key] = v

                signature_param_names = [
                    param.name
                    for param in inspect.signature(
                        builder_class.__init__
                    ).parameters.values()
                    if param.name != "self"
                ]
                builder_kwargs = {
                    k: v
                    for k, v in builder_kwargs.items()
                    if k in signature_param_names
                }
            return builder_class(**builder_kwargs)
        raise ValueError(f"Membership function builder '{name}' is not registered.")
