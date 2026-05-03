"""
PCA decomposition of yield curve movements into level, slope, curvature factors.

This is the foundation of the Monte Carlo risk engine.
The three factors explain ~85-95% of yield curve variance across history.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PCAModel:
    """
    Principal Component Analysis model for yield curve factor decomposition.

    Attributes:
        factor_loadings: shape (n_tenors, n_factors) — the eigenvectors
        eigenvalues: shape (n_factors,) — variance of each factor
        explained_variance: shape (n_factors,) — proportion of variance explained
        mean_changes: shape (n_tenors,) — mean daily curve change
        tenors: shape (n_tenors,) — tenor points used in fitting
    """

    factor_loadings: np.ndarray  # (n_tenors, n_factors)
    eigenvalues: np.ndarray      # (n_factors,)
    explained_variance: np.ndarray  # (n_factors,) - proportions summing to <= 1
    mean_changes: np.ndarray     # (n_tenors,)
    tenors: np.ndarray           # (n_tenors,)

    def project(self, daily_change: np.ndarray) -> np.ndarray:
        """
        Project a yield curve change onto the factors.

        Args:
            daily_change: shape (n_tenors,) — daily yield changes at each tenor

        Returns:
            shape (n_factors,) — factor scores (projections onto eigenvectors)
        """
        # Center by subtracting mean
        centered = daily_change - self.mean_changes

        # Project onto first n_factors eigenvectors
        scores = centered @ self.factor_loadings

        return scores

    def reconstruct(self, factor_scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct a curve change from factor scores.

        Args:
            factor_scores: shape (n_factors,) — factor scores

        Returns:
            shape (n_tenors,) — reconstructed yield curve change
        """
        # Reverse the projection: V @ scores gives back centered values
        reconstructed = self.factor_loadings @ factor_scores

        # Add back the mean
        return reconstructed + self.mean_changes

    def simulate_scenarios(
        self,
        n_scenarios: int,
        horizon_days: int = 1,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Generate random yield curve change scenarios using PCA factors.

        Algorithm:
        1. For each scenario:
           a. Generate standard normal random variables z ~ N(0,1) for each factor
           b. Scale by factor volatility: z * sqrt(eigenvalue)
           c. Account for time horizon: multiply by sqrt(horizon_days)
           d. Reconstruct curve change: V @ scaled_z

        Args:
            n_scenarios: Number of scenarios to generate
            horizon_days: Time horizon (for scaling volatility). Default 1 day for 1-day VaR
            random_state: Random seed for reproducibility

        Returns:
            shape (n_scenarios, n_tenors) — simulated curve changes
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_factors = self.factor_loadings.shape[1]
        n_tenors = self.factor_loadings.shape[0]

        scenarios = np.zeros((n_scenarios, n_tenors))

        # Standard normal draws for each factor, for each scenario
        z = np.random.standard_normal((n_scenarios, n_factors))  # (n_scenarios, n_factors)

        # Scale by factor volatility (standard deviation = sqrt(eigenvalue))
        # Also scale by sqrt(horizon) to account for multi-day VaR
        factor_vols = np.sqrt(self.eigenvalues * horizon_days)  # (n_factors,)
        scaled_z = z * factor_vols[np.newaxis, :]  # (n_scenarios, n_factors)

        # Reconstruct curve changes: each scenario is a linear combo of eigenvectors
        # V @ scores^T gives (n_tenors, n_scenarios), so transpose for (n_scenarios, n_tenors)
        curve_changes = (scaled_z @ self.factor_loadings.T)  # (n_scenarios, n_tenors)

        # Add mean
        scenarios = curve_changes + self.mean_changes[np.newaxis, :]

        return scenarios

    def get_explained_variance_ratio(self) -> float:
        """Total variance explained by the factors."""
        return float(self.explained_variance.sum())


def fit_pca(
    yield_history: np.ndarray,
    tenors: np.ndarray,
    n_factors: int = 3,
) -> PCAModel:
    """
    Fit PCA model to historical yield curve data.

    Args:
        yield_history: shape (n_days, n_tenors) — historical yield curves
        tenors: shape (n_tenors,) — tenor points
        n_factors: Number of factors to extract

    Returns:
        PCAModel with fitted parameters

    Raises:
        ValueError: If input dimensions don't match
    """
    if yield_history.shape[1] != len(tenors):
        raise ValueError(
            f"yield_history has {yield_history.shape[1]} tenors "
            f"but tenors array has {len(tenors)}"
        )

    if n_factors > yield_history.shape[1]:
        raise ValueError(f"Cannot extract {n_factors} factors from {yield_history.shape[1]} tenors")

    logger.info(f"Fitting PCA to {yield_history.shape[0]} days, {yield_history.shape[1]} tenors, {n_factors} factors")

    # 1. COMPUTE DAILY CHANGES
    # Only use complete pairs (skip first row which has no previous day)
    daily_changes = np.diff(yield_history, axis=0)  # shape (n_days-1, n_tenors)

    # 2. COMPUTE COVARIANCE MATRIX
    # Mean-center the changes
    mean_changes = daily_changes.mean(axis=0)  # shape (n_tenors,)
    centered_changes = daily_changes - mean_changes[np.newaxis, :]  # shape (n_days-1, n_tenors)

    # Covariance matrix
    cov_matrix = (centered_changes.T @ centered_changes) / (len(centered_changes) - 1)  # (n_tenors, n_tenors)

    # 3. EIGENDECOMPOSITION
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalue (descending)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # 4. EXTRACT FIRST n_factors
    factor_loadings = eigenvectors[:, :n_factors]  # shape (n_tenors, n_factors)
    factor_eigenvalues = eigenvalues[:n_factors]  # shape (n_factors,)

    # 5. COMPUTE EXPLAINED VARIANCE
    total_variance = eigenvalues.sum()
    explained_variance = factor_eigenvalues / total_variance  # (n_factors,)

    # Log summary
    logger.info(f"PCA factors explain {explained_variance.sum():.1%} of variance")
    logger.info(f"  PC1 (level):       {explained_variance[0]:.1%}")
    if len(explained_variance) > 1:
        logger.info(f"  PC2 (slope):       {explained_variance[1]:.1%}")
    if len(explained_variance) > 2:
        logger.info(f"  PC3 (curvature):   {explained_variance[2]:.1%}")

    # Interpret the factors (optional logging)
    _log_factor_interpretation(factor_loadings, tenors)

    model = PCAModel(
        factor_loadings=factor_loadings,
        eigenvalues=factor_eigenvalues,
        explained_variance=explained_variance,
        mean_changes=mean_changes,
        tenors=tenors,
    )

    return model


def _log_factor_interpretation(factor_loadings: np.ndarray, tenors: np.ndarray) -> None:
    """Log interpretation of PCA factors for debugging."""
    for i in range(min(3, factor_loadings.shape[1])):
        loadings = factor_loadings[:, i]

        # PC1 (level) has similar loading across all tenors
        # PC2 (slope) has opposite loadings (short vs long)
        # PC3 (curvature) has hump shape

        min_loading = loadings.min()
        max_loading = loadings.max()
        short_loading = loadings[0]
        long_loading = loadings[-1]

        factor_names = ["level", "slope", "curvature"]
        factor_name = factor_names[i] if i < 3 else f"factor_{i}"

        logger.debug(
            f"PC{i+1} ({factor_name}): "
            f"range=[{min_loading:.4f}, {max_loading:.4f}], "
            f"short={short_loading:.4f}, long={long_loading:.4f}"
        )
