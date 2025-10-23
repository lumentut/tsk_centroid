from .t1_gaussian_mf_ploter import T1GaussianMfPloter
from .t1_triangular_mf_ploter import T1TriangularMfPloter
from .t1_trapezoidal_mf_ploter import T1TrapezoidalMfPloter
from .t2_gaussian_mf_ploter import T2GaussianMfPloter
from .t2_triangular_mf_ploter import T2TriangularMfPloter
from .t2_trapezoidal_mf_ploter import T2TrapezoidalMfPloter
from src.fis.fuzzy_logic.mfs.mf_factory import MFType1, MFType2

MfPloter = (
    T1GaussianMfPloter
    | T1TriangularMfPloter
    | T2GaussianMfPloter
    | T2TriangularMfPloter
    | T1TrapezoidalMfPloter
    | T2TrapezoidalMfPloter
)


class MfPloterFactory:
    """Factory class for creating membership function ploters."""

    _ploters = {}

    @classmethod
    def register_ploter(cls, name: str, ploter_class: type):
        """Register a new membership function ploter.

        Args:
            name (str): The name of the membership function ploter.
            ploter_class (type): The membership function ploter class.
        """
        cls._ploters[name] = ploter_class

    @classmethod
    def create_ploter(cls, mf_type: str, **kwargs) -> MfPloter:
        """Create a membership function ploter based on the given type.

        Args:
            mf_type (str): The type of membership function ('t1_gaussian', 't1_triangular',
                          't2_gaussian', 't2_triangular').
            **kwargs: Additional arguments to pass to the ploter constructor.

        Returns:
            MfPloter: The created membership function ploter.

        Raises:
            ValueError: If the membership function type is not supported.
        """
        if mf_type not in cls._ploters:
            raise ValueError(
                f"Unsupported membership function type: {mf_type}. "
                f"Available types: {list(cls._ploters.keys())}"
            )

        ploter_class = cls._ploters[mf_type]
        return ploter_class(**kwargs)

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get a list of available membership function types.

        Returns:
            list[str]: List of available MF types.
        """
        return list(cls._ploters.keys())


MfPloterFactory.register_ploter(name=MFType1.GAUSSIAN, ploter_class=T1GaussianMfPloter)
MfPloterFactory.register_ploter(
    name=MFType1.TRIANGULAR, ploter_class=T1TriangularMfPloter
)
MfPloterFactory.register_ploter(
    name=MFType1.TRAPEZOIDAL, ploter_class=T1TrapezoidalMfPloter
)
MfPloterFactory.register_ploter(name=MFType2.GAUSSIAN, ploter_class=T2GaussianMfPloter)
MfPloterFactory.register_ploter(
    name=MFType2.TRIANGULAR, ploter_class=T2TriangularMfPloter
)
MfPloterFactory.register_ploter(
    name=MFType2.TRAPEZOIDAL, ploter_class=T2TrapezoidalMfPloter
)
