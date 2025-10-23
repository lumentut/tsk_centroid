from .ploters import MfPloterFactory


class MfPloterMixin:
    """Universal membership function plotter mixin that automatically detects MF types."""

    def plot_mfs_grid(
        self, mfs_per_row: int = 2, figsize: tuple[int, int] = (12, 12)
    ) -> None:
        """Plot membership functions in a grid layout using appropriate ploters.

        Args:
            mfs_per_row (int, optional): Number of plots per row. Defaults to 2.
            figsize (tuple[int, int], optional): Size of the figure. Defaults to (12, 12).
        """
        mfs_data = self.io_vars_.mfs_
        mf_type = mfs_data[list(mfs_data.keys())[0]][0]["type"]
        ploter = MfPloterFactory.create_ploter(
            mf_type, decimal_places=self.decimal_places
        )

        ploter.plot_mfs_grid(mfs_data, mfs_per_row, figsize)

    def plot_mf(self, feature_name: str, figsize: tuple[int, int] = (12, 6)) -> None:
        """Plot membership functions for a single feature using appropriate ploter.

        Args:
            feature_name (str): Name of the feature.
            figsize (tuple[int, int], optional): Size of the figure. Defaults to (12, 6).
        """
        mfs_data = self.io_vars_.mfs_
        mf_data = mfs_data[feature_name]

        if not mf_data:
            raise ValueError("No membership functions provided")

        # Detect the MF type from the first MF (assuming all are the same type)
        mf_type = mf_data[0]["type"]
        ploter = MfPloterFactory.create_ploter(
            mf_type, decimal_places=self.decimal_places
        )

        ploter.plot_mf(mf_data, feature_name, figsize)

    @classmethod
    def get_available_ploter_types(cls) -> list[str]:
        """Get a list of available ploter types."""
        return MfPloterFactory.get_available_types()
