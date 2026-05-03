"""DV01 calculation: dollar value of a basis point."""

import logging

import numpy as np

from greeks.duration import DurationCalculator
from pricing.cashflow_generator import CashFlowSchedule
from pricing.ytm_solver import solve_ytm
from pricing.bond_pricer import BondPricer
from curves.bootstrapper import SpotCurve

logger = logging.getLogger(__name__)


class DV01Calculator:
    """Computes DV01 for bonds."""

    @staticmethod
    def dv01_from_ytm(
        cashflows: CashFlowSchedule,
        market_price: float,
        notional: float,
        face_value: float = 100.0,
    ) -> float:
        """
        Compute DV01 from YTM.

        DV01 = ModD * Price/100 * Notional / 10000

        Units: rupees (or base currency) per basis point.

        Args:
            cashflows: Bond cash flows
            market_price: Bond price (as % of par, e.g., 102.50)
            notional: Notional amount in currency (e.g., Rs)
            face_value: Face value per bond (default 100)

        Returns:
            DV01 in currency units
        """
        # Solve for YTM
        ytm = solve_ytm(cashflows, market_price)
        if ytm is None:
            logger.warning(f"Could not solve YTM for price {market_price}, using 0 DV01")
            return 0.0

        # Compute modified duration
        mod_dur = DurationCalculator.modified_duration(cashflows, ytm)

        # DV01 = ModD * (Price/100) * (Notional/10000) * FaceValue
        # Simplified: DV01 = ModD * Price * Notional / 1e6
        # Or: DV01 = ModD * (market_price/100) * notional / 10000
        dv01 = mod_dur * (market_price / 100.0) * (notional / 10000.0)

        return dv01

    @staticmethod
    def dv01_from_curve(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
        notional: float = 1_000_000.0,  # Rs 10L
        bump_bps: float = 1.0,
    ) -> float:
        """
        Compute DV01 via numerical repricing (with curve bump).

        Algorithm:
        1. Price at current curve: P_0
        2. Price with spread bumped +1bp: P_up
        3. DV01 = |P_down - P_up| * (notional/100) / 2 / bump_bps

        Unfortunately, we can't directly bump the immutable SpotCurve.
        As a workaround, compute via spread sensitivity.

        Args:
            cashflows: Bond cash flows
            spot_curve: Spot curve
            spread_bps: Current spread
            notional: Notional amount
            bump_bps: Bump size for finite difference

        Returns:
            DV01 in currency units
        """
        # Current price
        p0 = BondPricer.price(cashflows, spot_curve, spread_bps)

        # Price with spread bumped
        p_up = BondPricer.price(cashflows, spot_curve, spread_bps + bump_bps)
        p_down = BondPricer.price(cashflows, spot_curve, spread_bps - bump_bps)

        # Price change per basis point
        price_change = (p_down - p_up) / (2.0 * bump_bps)

        # DV01 = price change * notional / 100
        dv01 = abs(price_change) * (notional / 100.0)

        logger.debug(f"DV01: {dv01:.2f} for notional {notional:.0f}")

        return dv01

    @staticmethod
    def portfolio_dv01(
        cashflows_list: list[CashFlowSchedule],
        market_prices: list[float],
        notionals: list[float],
    ) -> float:
        """
        Compute total portfolio DV01 (sum of individual DV01s).

        Args:
            cashflows_list: List of bond cash flows
            market_prices: List of bond prices
            notionals: List of notional amounts

        Returns:
            Total portfolio DV01
        """
        if len(cashflows_list) != len(market_prices) or len(cashflows_list) != len(notionals):
            raise ValueError("All lists must have same length")

        dv01s = [
            DV01Calculator.dv01_from_ytm(cf, price, notional)
            for cf, price, notional in zip(cashflows_list, market_prices, notionals)
        ]

        total_dv01 = sum(dv01s)
        return total_dv01

    @staticmethod
    def dv01_per_tenor(
        cashflows: CashFlowSchedule,
        market_price: float,
        notional: float,
    ) -> dict[str, float]:
        """
        Estimate DV01 contribution by maturity bucket.

        Rough approximation: allocate DV01 proportionally to coupon timing.

        Args:
            cashflows: Bond cash flows
            market_price: Bond price
            notional: Notional amount

        Returns:
            Dict mapping tenor bucket name to DV01 contribution
        """
        total_dv01 = DV01Calculator.dv01_from_ytm(cashflows, market_price, notional)

        if not cashflows.cash_flows:
            return {}

        # Allocate DV01 proportionally to PV-weighted cash flow timing
        dv01_by_tenor = {}
        total_pv = 0.0

        for cf in cashflows.cash_flows:
            ytm = solve_ytm(cashflows, market_price)
            if ytm is None:
                continue
            pv = cf.amount / ((1.0 + ytm) ** cf.tenor)
            total_pv += pv

        for cf in cashflows.cash_flows:
            ytm = solve_ytm(cashflows, market_price)
            if ytm is None:
                continue
            pv = cf.amount / ((1.0 + ytm) ** cf.tenor)

            weight = pv / total_pv if total_pv > 0 else 0

            tenor_bucket = f"{cf.tenor:.1f}Y"
            dv01_by_tenor[tenor_bucket] = total_dv01 * weight

        return dv01_by_tenor
