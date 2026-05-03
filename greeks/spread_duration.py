"""OAS spread duration for credit risk measurement."""

import logging

from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import BondPricer
from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


class SpreadDurationCalculator:
    """Computes spread duration (OAS duration) for corporate bonds."""

    @staticmethod
    def spread_duration(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        current_spread_bps: float,
        bump_bps: float = 1.0,
    ) -> float:
        """
        Compute spread duration: price sensitivity to 1bp spread change.

        Algorithm:
        1. Price at current spread: P_0
        2. Price at spread + 1bp: P_up
        3. Price at spread - 1bp: P_down
        4. SpreadDur = (P_down - P_up) / (2 * P_0 * 0.0001)

        This measures how much bond price changes per basis point of spread widening.

        Args:
            cashflows: Bond cash flows
            spot_curve: Spot curve (rates stay constant)
            current_spread_bps: Current OAS spread in basis points
            bump_bps: Bump size (default 1 bp)

        Returns:
            Spread duration (years)
        """
        # Price at current spread
        p0 = BondPricer.price(cashflows, spot_curve, current_spread_bps)

        if p0 == 0:
            logger.warning("Bond price is zero, cannot compute spread duration")
            return 0.0

        # Price at bumped spreads
        p_up = BondPricer.price(cashflows, spot_curve, current_spread_bps + bump_bps)
        p_down = BondPricer.price(cashflows, spot_curve, current_spread_bps - bump_bps)

        # Finite difference: (P_down - P_up) / (2 * P_0 * bump_bps/10000)
        dy = bump_bps / 10000.0
        spread_duration = (p_down - p_up) / (2 * p0 * dy)

        logger.debug(f"Spread duration: {spread_duration:.4f} years")

        return spread_duration

    @staticmethod
    def spread_dv01(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        current_spread_bps: float,
        notional: float,
        bump_bps: float = 1.0,
    ) -> float:
        """
        Compute spread DV01: rupees per 1bp spread move.

        Spread DV01 = SpreadDur * (Price/100) * (Notional / 10000)

        Args:
            cashflows: Bond cash flows
            spot_curve: Spot curve
            current_spread_bps: Current spread
            notional: Notional amount (e.g., Rs 1 crore)
            bump_bps: Bump size

        Returns:
            Spread DV01 in currency units
        """
        spread_duration = SpreadDurationCalculator.spread_duration(
            cashflows, spot_curve, current_spread_bps, bump_bps
        )

        price = BondPricer.price(cashflows, spot_curve, current_spread_bps)

        spread_dv01 = spread_duration * (price / 100.0) * (notional / 10000.0)

        logger.debug(f"Spread DV01: {spread_dv01:.2f}")

        return spread_dv01

    @staticmethod
    def rating_to_spread_sensitivity(rating: str) -> float:
        """
        Return typical spread sensitivity multiplier by rating.

        Used for approximate spread duration when full computation not available.

        Args:
            rating: Bond rating (AAA, AA, A, BBB, BB, B, CCC)

        Returns:
            Spread sensitivity factor (lower rating = higher sensitivity)
        """
        sensitivity_map = {
            "AAA": 0.8,
            "AA": 0.9,
            "A": 1.0,
            "BBB": 1.1,
            "BB": 1.3,
            "B": 1.5,
            "CCC": 2.0,
        }

        return sensitivity_map.get(rating, 1.0)

    @staticmethod
    def approx_spread_dv01_by_rating(
        cashflows: CashFlowSchedule,
        ytm: float,
        rating: str,
        notional: float,
    ) -> float:
        """
        Approximate spread DV01 using rating-based sensitivity.

        Useful when full repricing not available.

        Args:
            cashflows: Bond cash flows
            ytm: Yield to maturity
            rating: Bond rating
            notional: Notional amount

        Returns:
            Approximate spread DV01
        """
        from greeks.duration import DurationCalculator

        mod_duration = DurationCalculator.modified_duration(cashflows, ytm)
        sensitivity = SpreadDurationCalculator.rating_to_spread_sensitivity(rating)

        approx_spread_duration = mod_duration * sensitivity

        approx_spread_dv01 = approx_spread_duration * (notional / 10000.0)

        return approx_spread_dv01

    @staticmethod
    def portfolio_spread_dv01(
        cashflows_list: list[CashFlowSchedule],
        spot_curve: SpotCurve,
        spreads_bps: list[float],
        notionals: list[float],
    ) -> float:
        """
        Compute total portfolio spread DV01.

        Args:
            cashflows_list: List of bond cash flows
            spot_curve: Spot curve
            spreads_bps: List of spreads (one per bond)
            notionals: List of notionals (one per bond)

        Returns:
            Total portfolio spread DV01
        """
        total_spread_dv01 = 0.0

        for cf, spread, notional in zip(cashflows_list, spreads_bps, notionals):
            sdv01 = SpreadDurationCalculator.spread_dv01(cf, spot_curve, spread, notional)
            total_spread_dv01 += sdv01

        return total_spread_dv01
