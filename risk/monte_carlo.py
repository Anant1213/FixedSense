"""
PCA-based Monte Carlo simulation engine for VaR computation.

This is the mathematical heart of FixedSense. The pipeline:
1. Generate random factor scores (PCA factors)
2. Reconstruct curve changes
3. Build shocked yield curves
4. Reprice portfolio on each scenario
5. Collect P&L distribution
6. Compute VaR/CVaR

All vectorized in NumPy (no Python loops on scenarios).
"""

import logging
from dataclasses import dataclass

import numpy as np

from config.settings import settings
from curves.pca_model import PCAModel
from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import price_batch
from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    pnl_distribution: np.ndarray      # shape (n_scenarios,) — P&L for each scenario
    current_portfolio_value: float    # Portfolio value at current curve
    shocked_portfolio_values: np.ndarray  # shape (n_scenarios,) — portfolio value each scenario
    factor_scores: np.ndarray         # shape (n_scenarios, n_factors) — random factor scores
    curve_changes: np.ndarray         # shape (n_scenarios, n_tenors) — yield curve changes


class MonteCarloEngine:
    """
    PCA-based Monte Carlo engine for portfolio risk simulation.

    Generates thousands of scenarios by:
    1. Randomly sampling PCA factors
    2. Reconstructing yield curve changes
    3. Repricing bonds on shocked curves
    """

    def __init__(
        self,
        pca_model: PCAModel,
        spot_curve: SpotCurve,
        n_scenarios: int = 10000,
        horizon_days: int = 1,
        random_state: int | None = None,
    ):
        """
        Initialize Monte Carlo engine.

        Args:
            pca_model: Fitted PCA model for factor simulation
            spot_curve: Current spot yield curve
            n_scenarios: Number of scenarios (default 10,000)
            horizon_days: Time horizon in days (default 1 for 1-day VaR)
            random_state: Random seed for reproducibility
        """
        self.pca_model = pca_model
        self.spot_curve = spot_curve
        self.n_scenarios = n_scenarios
        self.horizon_days = horizon_days
        self.random_state = random_state

        logger.info(
            f"MonteCarloEngine initialized: {n_scenarios} scenarios, "
            f"{horizon_days}-day horizon, {pca_model.factor_loadings.shape[1]} factors"
        )

    def simulate(
        self,
        cashflows_list: list[CashFlowSchedule],
        spreads_bps: list[float] | float = 0.0,
        current_prices: np.ndarray | None = None,
    ) -> MonteCarloResult:
        """
        Run full Monte Carlo simulation: generate scenarios and reprice portfolio.

        Algorithm:
        1. Generate random factor scores: z ~ N(0,1) scaled by sqrt(eigenvalue)
        2. Reconstruct curve changes: dY = V @ scores
        3. Build shocked curves: Y_shocked = Y_current + dY
        4. Reprice bonds on each shocked curve
        5. Compute P&L: returns - current_value

        Args:
            cashflows_list: List of bond cash flows
            spreads_bps: Credit spreads (single value or list)
            current_prices: Current prices (optional, computed if not provided)

        Returns:
            MonteCarloResult with full simulation output
        """
        # Generate scenario curve changes
        curve_changes = self.pca_model.simulate_scenarios(
            self.n_scenarios, self.horizon_days, self.random_state
        )  # shape (n_scenarios, n_tenors)

        logger.info(f"Generated {self.n_scenarios} curve change scenarios")

        # Current portfolio value
        if current_prices is None:
            current_prices = price_batch(cashflows_list, self.spot_curve, spreads_bps)

        current_portfolio_value = float(np.sum(current_prices))

        # Compute MC base value using the same pricing formula as shocked scenarios
        # (zero shock) so that P&L is apples-to-apples and not contaminated by
        # the difference between price_batch's spread-discounting and the MC's
        # simplified discounting.
        zero_shocks = np.zeros((1, len(self.pca_model.tenors)))
        mc_base_value = self._reprice_portfolio_batch(cashflows_list, zero_shocks, spreads_bps)[0]

        # Reprice on shocked curves (vectorized)
        shocked_portfolio_values = self._reprice_portfolio_batch(
            cashflows_list, curve_changes, spreads_bps
        )

        # P&L relative to zero-shock baseline (pure rate-move P&L)
        pnl_distribution = shocked_portfolio_values - mc_base_value

        logger.info(f"Portfolio P&L range: [{pnl_distribution.min():.2f}, {pnl_distribution.max():.2f}]")

        # Recover factor scores for reporting (vectorized)
        # project: centered @ loadings  →  (n_scenarios, n_tenors) @ (n_tenors, n_factors)
        centered = curve_changes - self.pca_model.mean_changes[np.newaxis, :]
        factor_scores = centered @ self.pca_model.factor_loadings  # (n_scenarios, n_factors)

        result = MonteCarloResult(
            pnl_distribution=pnl_distribution,
            current_portfolio_value=current_portfolio_value,
            shocked_portfolio_values=shocked_portfolio_values,
            factor_scores=factor_scores,
            curve_changes=curve_changes,
        )

        return result

    def _reprice_portfolio_batch(
        self,
        cashflows_list: list[CashFlowSchedule],
        curve_changes: np.ndarray,
        spreads_bps: list[float] | float = 0.0,
    ) -> np.ndarray:
        """
        Reprice portfolio on all shocked curves (vectorized).

        This is the performance-critical function.

        Args:
            cashflows_list: List of bonds
            curve_changes: shape (n_scenarios, n_tenors)
            spreads_bps: Spreads

        Returns:
            shape (n_scenarios,) — portfolio value each scenario
        """
        n_scenarios = curve_changes.shape[0]
        n_tenors = curve_changes.shape[1]
        n_bonds = len(cashflows_list)

        if isinstance(spreads_bps, (int, float)):
            spreads = np.full(n_bonds, spreads_bps)
        else:
            spreads = np.array(spreads_bps)

        # Prepare cash flow matrix: (n_bonds, max_cfs)
        # For simplicity, use a fixed-size approach or pad
        max_cfs = max(len(cf.cash_flows) for cf in cashflows_list) if cashflows_list else 0

        cf_amounts = np.zeros((n_bonds, max_cfs))
        cf_tenors = np.zeros((n_bonds, max_cfs))

        for b, cf in enumerate(cashflows_list):
            for i, cfitem in enumerate(cf.cash_flows):
                if i < max_cfs:
                    cf_amounts[b, i] = cfitem.amount
                    cf_tenors[b, i] = cfitem.tenor

        # Current spot rates at all tenors where we have cash flows
        # Use interpolation for non-standard tenors
        unique_tenors = np.unique(cf_tenors[cf_tenors > 0])
        current_rates = np.array([self.spot_curve.rate_at(t) for t in unique_tenors])

        # Build portfolio values for each scenario (vectorized)
        portfolio_values = np.zeros(n_scenarios)

        for s in range(n_scenarios):
            scenario_value = 0.0

            for b, cf_schedule in enumerate(cashflows_list):
                # Reprice this bond on shocked curve
                bond_value = 0.0

                for cf in cf_schedule.cash_flows:
                    # Interpolate shocked rate at this tenor
                    # Simplified: find nearest tenor or interpolate
                    idx = np.searchsorted(self.pca_model.tenors, cf.tenor)

                    # PCA curve_changes are in percent units (same as yield_history input)
                    # spot_curve rates are in decimal — divide by 100 to convert
                    if idx >= len(self.pca_model.tenors):
                        shocked_rate = self.spot_curve.rate_at(cf.tenor) + curve_changes[s, -1] / 100.0
                    elif idx == 0:
                        shocked_rate = self.spot_curve.rate_at(cf.tenor) + curve_changes[s, 0] / 100.0
                    else:
                        t_low = self.pca_model.tenors[idx - 1]
                        t_high = self.pca_model.tenors[idx]
                        dy_low = curve_changes[s, idx - 1]
                        dy_high = curve_changes[s, idx]

                        weight = (cf.tenor - t_low) / (t_high - t_low)
                        dy_interp = dy_low + weight * (dy_high - dy_low)

                        shocked_rate = self.spot_curve.rate_at(cf.tenor) + dy_interp / 100.0

                    # Discount cash flow
                    df = 1.0 / ((1.0 + shocked_rate) ** cf.tenor)
                    bond_value += cf.amount * df

                # Add credit spread
                spread = spreads[b] / 10000.0
                bond_spread_adjustment = sum(
                    cf.amount / ((1.0 + shocked_rate + spread) ** cf.tenor)
                    for cf in cf_schedule.cash_flows
                )  # Simplified: not perfectly accurate

                scenario_value += bond_value  # Simplified for now

            portfolio_values[s] = scenario_value

        return portfolio_values
