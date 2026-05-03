"""Bond pricer using discounted cash flows."""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np

from curves.bootstrapper import SpotCurve
from pricing.cashflow_generator import CashFlowSchedule, DayCountMethod, _tenor_in_years

logger = logging.getLogger(__name__)


@dataclass
class BondPricingResult:
    """Result of bond pricing computation."""

    bond_id: str
    clean_price: float      # Price without accrued interest
    dirty_price: float      # Price with accrued interest
    accrued_interest: float # Accrued coupon since last payment
    ytm: float | None       # Yield to maturity (if computed)


class BondPricer:
    """
    Bond pricer using spot curve and discounted cash flows.

    Price = sum(CF_i * DF(t_i)) for all cash flows
    where DF(t) = 1 / (1 + z(t))^t is the discount factor

    For corporate bonds, add credit spread:
    Effective rate = z(t) + spread
    """

    @staticmethod
    def price(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
    ) -> float:
        """
        Compute dirty price (with accrued interest) of a bond.

        Args:
            cashflows: Cash flow schedule
            spot_curve: Spot yield curve for discounting
            spread_bps: Credit spread in basis points (default 0 for government bonds)

        Returns:
            Bond price as percentage of par (100 for par bond)
        """
        spread = spread_bps / 10000.0  # Convert bps to decimal

        price = 0.0
        for cf in cashflows.cash_flows:
            # Get spot rate at this tenor
            z = spot_curve.rate_at(cf.tenor)

            # Effective rate = spot + spread
            effective_rate = z + spread

            # Discount factor
            df = 1.0 / ((1.0 + effective_rate) ** cf.tenor)

            # Add discounted cash flow
            price += cf.amount * df

        return price

    @staticmethod
    def accrued_interest(
        cashflows: CashFlowSchedule,
        as_of_date: date | None = None,
        day_count_method: DayCountMethod = DayCountMethod.ACT_365,
    ) -> float:
        """
        Compute accrued interest on a bond.

        Accrued interest = coupon * (days_since_last_coupon / days_in_coupon_period)

        Args:
            cashflows: Cash flow schedule
            as_of_date: Date for which to compute accrual (default today)
            day_count_method: Day count convention

        Returns:
            Accrued interest amount in currency (e.g., Rs)
        """
        if as_of_date is None:
            as_of_date = date.today()

        if not cashflows.cash_flows:
            return 0.0

        # Find the last coupon date and next coupon date
        last_coupon_date = None
        next_coupon_date = None

        for cf in cashflows.cash_flows:
            if cf.date <= as_of_date:
                last_coupon_date = cf.date
            elif cf.date > as_of_date:
                next_coupon_date = cf.date
                break

        if last_coupon_date is None or next_coupon_date is None:
            # No accrual if bond hasn't started or has matured
            return 0.0

        # Days since last coupon
        days_since = (as_of_date - last_coupon_date).days

        # Days in coupon period
        days_in_period = (next_coupon_date - last_coupon_date).days

        # Coupon per period
        coupon_per_period = (cashflows.face_value * cashflows.coupon_rate) / cashflows.coupon_frequency

        # Accrued = coupon * (days_since / days_in_period)
        accrued = coupon_per_period * (days_since / days_in_period)

        return accrued

    @staticmethod
    def clean_price(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
        as_of_date: date | None = None,
    ) -> float:
        """
        Compute clean price (without accrued interest).

        clean_price = dirty_price - accrued_interest

        Args:
            cashflows: Cash flow schedule
            spot_curve: Spot curve
            spread_bps: Credit spread in bps
            as_of_date: Valuation date (default today)

        Returns:
            Clean price
        """
        dirty = BondPricer.price(cashflows, spot_curve, spread_bps)
        accrued = BondPricer.accrued_interest(cashflows, as_of_date)

        return dirty - accrued

    @staticmethod
    def pv_basis_point(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
        bump_bps: float = 1.0,
    ) -> float:
        """
        Compute price value of a basis point (PVBP).

        PVBP = P_down - P_up for a 1bp parallel shift
        This is an alternative to duration-based calculation.

        Args:
            cashflows: Cash flow schedule
            spot_curve: Current spot curve
            spread_bps: Current spread in bps
            bump_bps: Bump size (default 1 bp)

        Returns:
            Price change per basis point (in currency units)
        """
        # Current price
        p0 = BondPricer.price(cashflows, spot_curve, spread_bps)

        # Can't easily bump the spot curve object (immutable), so estimate via duration
        # This is a simplified version; full version would rebuild the curve
        # For now, use numerical approximation

        # Approximate: use small yield change
        shift = bump_bps / 10000.0  # 1bp to decimal

        # Shift all rates by the bump amount (simplified parallel shift)
        # This requires recomputing tenors or using duration

        # Simplified: use average spread-based sensitivity
        p_bumped = BondPricer.price(cashflows, spot_curve, spread_bps + bump_bps)
        p_original = BondPricer.price(cashflows, spot_curve, spread_bps)

        pvbp = abs(p_bumped - p_original)
        return pvbp


def price_batch(
    cashflows_list: list[CashFlowSchedule],
    spot_curve: SpotCurve,
    spreads_bps: list[float] | float = 0.0,
) -> np.ndarray:
    """
    Price multiple bonds in batch (vectorized).

    Args:
        cashflows_list: List of cash flow schedules
        spot_curve: Single spot curve for all bonds
        spreads_bps: Single spread or list of spreads (one per bond)

    Returns:
        Array of prices (length = len(cashflows_list))
    """
    if isinstance(spreads_bps, (int, float)):
        spreads = np.full(len(cashflows_list), spreads_bps)
    else:
        spreads = np.array(spreads_bps)

    prices = np.array(
        [BondPricer.price(cf, spot_curve, spread) for cf, spread in zip(cashflows_list, spreads)]
    )

    return prices
