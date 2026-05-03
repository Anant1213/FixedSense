"""Yield curve interpolation methods: cubic spline and Nelson-Siegel."""

import logging

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class CubicSplineInterpolator:
    """Cubic spline interpolation of yield curve."""

    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        """
        Initialize with observed tenor points.

        Args:
            tenors: Array of tenors in years
            rates: Array of rates (as decimals)
        """
        self.tenors = tenors
        self.rates = rates
        self.cs = CubicSpline(tenors, rates, bc_type="natural")
        logger.debug(f"CubicSpline fitted on {len(tenors)} points")

    def rate_at(self, tenor: float) -> float:
        """Get interpolated rate at tenor."""
        return float(self.cs(tenor))

    def rates_at(self, tenors: np.ndarray) -> np.ndarray:
        """Get interpolated rates at multiple tenors (vectorized)."""
        return np.asarray(self.cs(tenors))


class NelsonSiegelModel:
    """
    Nelson-Siegel parametric yield curve model.

    y(t) = β0 + β1*(1-exp(-t/τ))/(t/τ) + β2*((1-exp(-t/τ))/(t/τ) - exp(-t/τ))

    Parameters:
    - β0: Long-term yield level
    - β1: Slope (short-to-long difference)
    - β2: Curvature (hump)
    - τ: Decay parameter
    """

    def __init__(self):
        """Initialize with no parameters (fit required)."""
        self.beta0: float | None = None
        self.beta1: float | None = None
        self.beta2: float | None = None
        self.tau: float | None = None

    def _model_function(self, t: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
        """Evaluate Nelson-Siegel model."""
        beta0, beta1, beta2, tau = params

        # Avoid division by zero
        t_over_tau = np.where(t > 1e-6, t / tau, 1e-6)

        # Component 1: level
        level = beta0

        # Component 2: slope
        slope_factor = (1.0 - np.exp(-t_over_tau)) / t_over_tau
        slope = beta1 * slope_factor

        # Component 3: curvature
        curvature_factor = slope_factor - np.exp(-t_over_tau)
        curvature = beta2 * curvature_factor

        return level + slope + curvature

    def fit(self, tenors: np.ndarray, rates: np.ndarray) -> None:
        """
        Fit Nelson-Siegel parameters to observed yields.

        Uses scipy.optimize.minimize with initial guesses based on data.

        Args:
            tenors: Array of tenors (years)
            rates: Array of corresponding yields (decimals)
        """
        # Initial guesses based on data characteristics
        beta0_init = rates[-1]  # Long-term level = long end rate
        beta1_init = rates[0] - rates[-1]  # Slope = short - long
        beta2_init = 0.0  # Curvature offset
        tau_init = 1.0  # Typical decay parameter

        def objective(params: np.ndarray) -> float:
            """Sum of squared errors."""
            predicted = self._model_function(tenors, params)
            sse = np.sum((predicted - rates) ** 2)
            return sse

        # Bounds: beta0 and beta1 unconstrained, beta2 unconstrained, tau > 0
        bounds = [
            (0.0001, 0.50),  # beta0
            (-0.50, 0.50),   # beta1
            (-0.50, 0.50),   # beta2
            (0.01, 10.0),    # tau
        ]

        result = minimize(
            objective,
            x0=[beta0_init, beta1_init, beta2_init, tau_init],
            method="L-BFGS-B",
            bounds=bounds,
        )

        if not result.success:
            logger.warning(f"Nelson-Siegel fit may not have converged: {result.message}")

        self.beta0, self.beta1, self.beta2, self.tau = result.x
        logger.debug(
            f"Nelson-Siegel fitted: β0={self.beta0:.6f}, β1={self.beta1:.6f}, β2={self.beta2:.6f}, τ={self.tau:.6f}"
        )

    def rate_at(self, tenor: float) -> float:
        """Get rate at tenor."""
        if self.beta0 is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        params = (self.beta0, self.beta1, self.beta2, self.tau)
        return float(self._model_function(np.array([tenor]), params)[0])

    def rates_at(self, tenors: np.ndarray) -> np.ndarray:
        """Get rates at multiple tenors (vectorized)."""
        if self.beta0 is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        params = (self.beta0, self.beta1, self.beta2, self.tau)
        return self._model_function(tenors, params)

    def get_params(self) -> dict[str, float]:
        """Get fitted parameters."""
        if self.beta0 is None:
            raise ValueError("Model not fitted yet.")

        return {
            "beta0": self.beta0,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "tau": self.tau,
        }


def interpolate_cubic(
    tenors: np.ndarray,
    rates: np.ndarray,
    query_tenors: np.ndarray,
) -> np.ndarray:
    """
    Interpolate yields using cubic spline.

    Args:
        tenors: Known tenor points
        rates: Known rates
        query_tenors: Tenors to interpolate

    Returns:
        Interpolated rates
    """
    cs = CubicSplineInterpolator(tenors, rates)
    return cs.rates_at(query_tenors)


def interpolate_nelson_siegel(
    tenors: np.ndarray,
    rates: np.ndarray,
    query_tenors: np.ndarray,
) -> np.ndarray:
    """
    Interpolate yields using Nelson-Siegel model.

    Args:
        tenors: Known tenor points
        rates: Known rates
        query_tenors: Tenors to interpolate

    Returns:
        Interpolated rates
    """
    model = NelsonSiegelModel()
    model.fit(tenors, rates)
    return model.rates_at(query_tenors)
