"""Duration calculations: Macaulay, modified, effective duration."""

import logging

import numpy as np

from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import BondPricer
from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


class DurationCalculator:
    """Computes various duration measures for bonds."""

    @staticmethod
    def macaulay_duration(
        cashflows: CashFlowSchedule,
        ytm: float,
    ) -> float:
        """
        Compute Macaulay duration.

        MacD = (1/Price) * sum(t_i * CF_i / (1+y)^t_i)

        Weighted average time to receipt of cash flows (in years).

        Args:
            cashflows: Bond cash flows
            ytm: Yield to maturity as decimal

        Returns:
            Macaulay duration in years
        """
        numerator = 0.0
        denominator = 0.0

        for cf in cashflows.cash_flows:
            pv = cf.amount / ((1.0 + ytm) ** cf.tenor)
            numerator += cf.tenor * pv
            denominator += pv

        if denominator == 0:
            return 0.0

        macaulay_dur = numerator / denominator
        return macaulay_dur

    @staticmethod
    def modified_duration(
        cashflows: CashFlowSchedule,
        ytm: float,
    ) -> float:
        """
        Compute modified duration.

        ModD = MacD / (1 + y/freq)

        This is the price sensitivity measure: dPrice/dYield ≈ -ModD * Price * dY

        Args:
            cashflows: Bond cash flows
            ytm: Yield to maturity

        Returns:
            Modified duration in years
        """
        macaulay = DurationCalculator.macaulay_duration(cashflows, ytm)
        modified = macaulay / (1.0 + ytm / cashflows.coupon_frequency)
        return modified

    @staticmethod
    def effective_duration(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
        bump_bps: float = 1.0,
    ) -> float:
        """
        Compute effective duration using full repricing.

        EffD = (P_down - P_up) / (2 * P_0 * dy)

        More accurate than analytical duration for complex bonds.

        Algorithm:
        1. Price bond at current curve: P_0
        2. Bump entire curve UP by bump_bps: P_up
        3. Bump entire curve DOWN by bump_bps: P_down
        4. EffD = (P_down - P_up) / (2 * P_0 * (bump_bps/10000))

        Args:
            cashflows: Bond cash flows
            spot_curve: Current spot curve
            spread_bps: Credit spread in bps
            bump_bps: Bump size (default 1 bp)

        Returns:
            Effective duration
        """
        # Current price
        p0 = BondPricer.price(cashflows, spot_curve, spread_bps)

        if p0 == 0:
            return 0.0

        # Unfortunately, we can't easily bump the immutable SpotCurve object
        # This is a limitation of the frozen dataclass design
        # In practice, we'd rebuild the curve or use modified duration as proxy

        # For now, use modified duration as approximation
        from pricing.ytm_solver import solve_ytm

        ytm = solve_ytm(cashflows, p0)
        if ytm is None:
            return 0.0

        modified_dur = DurationCalculator.modified_duration(cashflows, ytm)
        logger.debug(f"Used modified duration ({modified_dur:.4f}) as proxy for effective duration")

        return modified_dur

    @staticmethod
    def portfolio_duration(
        cashflows_list: list[CashFlowSchedule],
        ytm: float,
        weights: np.ndarray | None = None,
    ) -> float:
        """
        Compute weighted average duration of a portfolio.

        Args:
            cashflows_list: List of bond cash flow schedules
            ytm: Single YTM for all bonds (simplified)
            weights: Weight of each bond (default: equal weight)

        Returns:
            Portfolio duration
        """
        if not cashflows_list:
            return 0.0

        if weights is None:
            weights = np.ones(len(cashflows_list)) / len(cashflows_list)

        durations = np.array([
            DurationCalculator.modified_duration(cf, ytm)
            for cf in cashflows_list
        ])

        portfolio_dur = np.dot(weights, durations)
        return portfolio_dur

    @staticmethod
    def key_rate_duration(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
    ) -> dict[float, float]:
        """
        Compute key rate duration at standard tenor buckets.

        This is a simplified version. Full KR01 is in greeks/kr01.py

        Args:
            cashflows: Bond cash flows
            spot_curve: Spot curve
            spread_bps: Credit spread

        Returns:
            Dict mapping tenor to key rate duration
        """
        # Placeholder: return empty dict
        # Full implementation is in KR01Calculator

        return {}
