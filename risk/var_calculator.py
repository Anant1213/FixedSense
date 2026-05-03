"""VaR computation: parametric, historical, Monte Carlo methods."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from risk.monte_carlo import MonteCarloResult

logger = logging.getLogger(__name__)


@dataclass
class VaRReport:
    """VaR computation result."""

    parametric_var: float
    historical_var: float | None
    monte_carlo_var: float
    confidence_level: float
    portfolio_value: float
    var_as_pct_of_portfolio: float
    z_score: float


class VaRCalculator:
    """Computes Value at Risk using multiple methods."""

    @staticmethod
    def parametric_var(
        portfolio_value: float,
        portfolio_vol: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Parametric VaR assuming normal distribution.

        VaR = portfolio_value * z_score * vol * sqrt(horizon)

        Args:
            portfolio_value: Current portfolio value
            portfolio_vol: Portfolio volatility (daily, as decimal)
            confidence: Confidence level (default 95%)
            horizon_days: Time horizon in days (default 1)

        Returns:
            VaR in currency units
        """
        z_score = stats.norm.ppf(confidence)  # 1.645 for 95%, 2.326 for 99%

        var = portfolio_value * z_score * portfolio_vol * np.sqrt(horizon_days)

        return var

    @staticmethod
    def historical_var(
        returns: np.ndarray,
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> float:
        """
        Historical VaR: use percentile of historical returns.

        VaR = -percentile(returns, (1-confidence)*100) * portfolio_value

        No distributional assumptions.

        Args:
            returns: Array of historical returns (as decimals)
            portfolio_value: Current portfolio value
            confidence: Confidence level

        Returns:
            VaR in currency units
        """
        loss_percentile = (1 - confidence) * 100  # e.g., 5 for 95% confidence
        worst_loss = -np.percentile(returns, loss_percentile)

        var = worst_loss * portfolio_value

        return var

    @staticmethod
    def monte_carlo_var(
        mc_result: MonteCarloResult,
        confidence: float = 0.95,
    ) -> float:
        """
        Monte Carlo VaR: percentile of simulated P&L distribution.

        VaR = -percentile(pnl_distribution, (1-confidence)*100)

        This uses the full PCA-based simulation results.

        Args:
            mc_result: MonteCarloResult from simulation
            confidence: Confidence level

        Returns:
            VaR in currency units (absolute value, loss magnitude)
        """
        # VaR = loss exceeded by only (1-confidence) of scenarios
        # = confidence-th percentile of the loss distribution
        losses = -mc_result.pnl_distribution  # positive = loss
        var = np.percentile(losses, confidence * 100)

        return var

    @staticmethod
    def compute_all_var(
        portfolio_value: float,
        portfolio_vol: float,
        mc_result: MonteCarloResult | None = None,
        historical_returns: np.ndarray | None = None,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> VaRReport:
        """
        Compute VaR using all available methods and return comparison.

        Args:
            portfolio_value: Current portfolio value
            portfolio_vol: Portfolio volatility
            mc_result: Monte Carlo results (optional)
            historical_returns: Historical returns (optional)
            confidence: Confidence level
            horizon_days: Time horizon

        Returns:
            VaRReport with all three VaR estimates
        """
        # Parametric VaR (always available)
        param_var = VaRCalculator.parametric_var(
            portfolio_value, portfolio_vol, confidence, horizon_days
        )

        # Historical VaR (if data provided)
        hist_var = None
        if historical_returns is not None and len(historical_returns) > 0:
            hist_var = VaRCalculator.historical_var(historical_returns, portfolio_value, confidence)

        # Monte Carlo VaR (if simulation provided)
        mc_var = mc_result.pnl_distribution.std() * portfolio_value * 2.0  # Rough estimate
        if mc_result is not None:
            mc_var = VaRCalculator.monte_carlo_var(mc_result, confidence)
        else:
            mc_var = param_var  # Fallback

        z_score = stats.norm.ppf(confidence)

        report = VaRReport(
            parametric_var=param_var,
            historical_var=hist_var,
            monte_carlo_var=mc_var,
            confidence_level=confidence,
            portfolio_value=portfolio_value,
            var_as_pct_of_portfolio=(mc_var / portfolio_value) * 100,
            z_score=z_score,
        )

        return report

    @staticmethod
    def var_for_multiple_confidence_levels(
        mc_result: MonteCarloResult,
        confidence_levels: list[float] = [0.90, 0.95, 0.99],
    ) -> dict[float, float]:
        """
        Compute VaR at multiple confidence levels from single MC simulation.

        Args:
            mc_result: Monte Carlo result
            confidence_levels: List of confidence levels

        Returns:
            Dict mapping confidence level to VaR
        """
        var_dict = {}

        for conf in confidence_levels:
            var = VaRCalculator.monte_carlo_var(mc_result, confidence=conf)
            var_dict[conf] = var

        return var_dict

    @staticmethod
    def var_contribution_by_bond(
        cashflows_list: list,
        notionals: np.ndarray,
        mc_result: MonteCarloResult,
        confidence: float = 0.95,
    ) -> dict:
        """
        Estimate marginal VaR contribution by bond (simplified).

        Args:
            cashflows_list: List of bonds
            notionals: Notional amounts
            mc_result: Monte Carlo result
            confidence: Confidence level

        Returns:
            Dict mapping bond ID to estimated marginal VaR
        """
        total_var = VaRCalculator.monte_carlo_var(mc_result, confidence)

        # Simplified: allocate VaR proportionally to notional
        total_notional = notionals.sum()

        var_by_bond = {}
        for i, cf in enumerate(cashflows_list):
            estimated_var = total_var * (notionals[i] / total_notional)
            var_by_bond[cf.bond_id] = estimated_var

        return var_by_bond
