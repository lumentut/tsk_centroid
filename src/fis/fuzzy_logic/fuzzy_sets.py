from pyit2fls import (
    T1FS,
    IT2FS,
    IT2FS_Gaussian_UncertStd,
    tri_mf,
    gaussian_mf,
    trapezoid_mf,
)
from .io_variable import MembershipFunction
from .mfs import MFType1, MFType2
from src.utils.domain import calculate_range


class FuzzySets:
    """
    A class to handle fuzzy sets and their membership functions.
    """

    def __init__(self, decimal_places: int = 4):
        self.decimal_places = decimal_places

    def _get_domain(self, mf: MembershipFunction) -> list[float]:
        """
        Get the domain for a given input variable.
        """
        domain_min = mf["universe"][0]
        domain_max = mf["universe"][1]
        return calculate_range(domain_min, domain_max, self.decimal_places)

    def _triangular_params(self, params):
        """Validate and fix triangular parameters to prevent division by zero.

        Args:
            params (list): A list containing the parameters [a, b, c].

        Returns:
            list: A list containing the validated parameters [a, b, c].
        """
        a, b, c = params[0], params[1], params[2]
        # Ensure a < b < c to prevent division by zero
        if a >= b:
            b = a + 0.01
        if b >= c:
            c = b + 0.01
        return [a, b, c]

    def _trapezoidal_params(self, params):
        """Validate and fix trapezoidal parameters to prevent invalid configurations.

        Args:
            params (list): A list containing the parameters [a, b, c, d].
        Returns:
            list: A list containing the validated parameters [a, b, c, d].
        """
        a, b, c, d = params[0], params[1], params[2], params[3]
        # Ensure a < b < c < d to prevent invalid trapezoidal shape
        if a >= b:
            b = a + 0.01
        if b >= c:
            c = b + 0.01
        if c >= d:
            d = c + 0.01
        return [a, b, c, d]

    def _gaussian_params(self, params):
        """Validate and fix Gaussian parameters to prevent invalid configurations.

        Args:
            params (list): A list containing the parameters [center, std].

        Returns:
            list: A list containing the validated parameters [center, std].
        """
        center, std = params[0], params[1]
        if std <= 0:
            std = 0.1
        return [std, center]

    def defined_by_mf(self, mf: MembershipFunction) -> T1FS | IT2FS:
        """
        Get the Fuzzy Sets for a given membership function.
        """
        domain = self._get_domain(mf)
        parameters = mf["parameters"]
        mf_type = mf["type"]

        if mf_type == MFType1.TRIANGULAR:
            mf_params = parameters["mf"]
            if len(mf_params) >= 3:
                mf_params = self._triangular_params(mf_params[:3])
                return T1FS(domain, mf=tri_mf, params=mf_params + [1.0])

        elif mf_type == MFType1.GAUSSIAN:
            mf_params = parameters["mf"]
            if len(mf_params) >= 2:
                mf_params = self._gaussian_params(mf_params[:2])
                return T1FS(domain, mf=gaussian_mf, params=mf_params + [1.0])

        elif mf_type == MFType1.TRAPEZOIDAL:
            mf_params = parameters["mf"]
            if len(mf_params) >= 4:
                mf_params = self._trapezoidal_params(mf_params[:4])
                return T1FS(domain, mf=trapezoid_mf, params=mf_params + [1.0])

        elif mf_type == MFType2.TRIANGULAR:
            lmf_params = parameters["lmf"]
            umf_params = parameters["umf"]

            if len(lmf_params) >= 3 and len(umf_params) >= 3:
                lmf_params = self._triangular_params(lmf_params[:3])
                umf_params = self._triangular_params(umf_params[:3])

                return IT2FS(
                    domain,
                    umf=tri_mf,
                    umf_params=umf_params + [1.0],
                    lmf=tri_mf,
                    lmf_params=lmf_params + [1.0],
                )

        elif mf_type == MFType2.GAUSSIAN:
            mean = parameters["mean"]
            sigma_upper = parameters["sigma_upper"]
            sigma_lower = parameters["sigma_lower"]

            return IT2FS_Gaussian_UncertStd(
                domain,
                [
                    mean,
                    (sigma_upper + sigma_lower) / 2,  # std_center,
                    abs(sigma_upper - sigma_lower),  # std_spread,
                    1,
                ],
            )

        elif mf_type == MFType2.TRAPEZOIDAL:
            lmf_params = self._trapezoidal_params(parameters["lmf"])
            umf_params = self._trapezoidal_params(parameters["umf"])

            if len(lmf_params) >= 4 and len(umf_params) >= 4:
                return IT2FS(
                    domain,
                    umf=trapezoid_mf,
                    umf_params=umf_params[:4] + [1.0],
                    lmf=trapezoid_mf,
                    lmf_params=lmf_params[:4] + [1.0],
                )

        else:
            raise ValueError(f"Unsupported MF type: {mf_type}")
