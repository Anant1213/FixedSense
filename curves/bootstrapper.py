"""Spot curve bootstrapping from par yields using Brentq root-finding."""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpotCurve:
    """
    Zero-coupon (spot) yield curve.

    Immutable (frozen dataclass) to prevent accidental mutation in computations.
    """

    tenors: np.ndarray
    spot_rates: np.ndarray
    as_of_date: date

    def rate_at(self, tenor: float) -> float:
        """
        Get interpolated spot rate at any tenor using linear interpolation.

        Linear interpolation is used (not cubic spline) to guarantee:
        - No oscillation between knots
        - Consistency with the bootstrapping algorithm
        - Monotone discount factors for upward-sloping curves
        """
        return float(np.interp(tenor, self.tenors, self.spot_rates))

    def discount_factor(self, tenor: float) -> float:
        """
        Get discount factor at tenor.

        DF = 1 / (1 + z(t))^t
        """
        z = self.rate_at(tenor)
        return 1.0 / ((1.0 + z) ** tenor)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Implied forward rate between two tenors.

        f(t1,t2) = ((1+z(t2))^t2 / (1+z(t1))^t1)^(1/(t2-t1)) - 1
        """
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be > t1 ({t1})")

        z1 = self.rate_at(t1)
        z2 = self.rate_at(t2)

        ratio = ((1.0 + z2) ** t2) / ((1.0 + z1) ** t1)
        forward = ratio ** (1.0 / (t2 - t1)) - 1.0

        return forward


def _solve_spot_rate(
    tenor: float,
    par_yield: float,
    coupon_freq: int,
    interp_curve: "SpotCurve | None",
) -> float:
    """
    Solve for the spot rate at `tenor` given the par yield and a curve for
    discounting intermediate coupons.

    When interp_curve is None (bootstrapping the very first tenor), assumes
    a flat yield curve equal to z everywhere — this gives the YTM equation.

    Returns the zero-coupon spot rate as a decimal.
    """
    dollar_coupon = (par_yield / coupon_freq) * 100.0  # e.g. $1.75 for 3.5% semi-annual
    n_periods = int(round(tenor * coupon_freq))

    # For tenors shorter than one coupon period (e.g. 1-month with semi-annual coupons),
    # there are no intermediate coupon dates.  Treat as a discount instrument:
    # price = (100 + par_yield × tenor × 100) / (1+z)^tenor = 100  →  z ≈ par_yield.
    if n_periods == 0:
        def _discount_eq(z: float) -> float:
            face_plus_interest = 100.0 * (1.0 + par_yield * tenor)
            return face_plus_interest / ((1.0 + z) ** tenor) - 100.0
        return brentq(_discount_eq, 1e-6, 0.80, xtol=1e-10)

    def par_bond_price(z: float) -> float:
        pv = 0.0
        for k in range(1, n_periods + 1):
            t = k / coupon_freq
            if k < n_periods:
                # Intermediate coupon: use interpolated known spot rates.
                # If no curve available yet (i=0), assume flat at z.
                rate = interp_curve.rate_at(t) if interp_curve is not None else z
                pv += dollar_coupon / ((1.0 + rate) ** t)
            else:
                # Final payment: coupon + $100 principal at unknown spot rate z
                pv += (dollar_coupon + 100.0) / ((1.0 + z) ** t)
        return pv - 100.0

    return brentq(par_bond_price, 1e-6, 0.80, xtol=1e-10)


def bootstrap_spot_curve(
    par_yields: dict[float, float],
    as_of_date: date = None,
) -> "SpotCurve":
    """
    Bootstrap zero-coupon spot curve from par bond yields.

    Two-pass algorithm for consistency with the final interpolated curve:

    Pass 1 — sequential bootstrapping:
        For each tenor i, solve for z(i) using flat/linear extrapolation of
        already-known spot rates for intermediate coupon discount rates.

    Pass 2 — refinement:
        Re-solve each z(i) using the FULL pass-1 curve (via rate_at()) to
        discount intermediate coupons. This eliminates the inconsistency
        between bootstrapping and pricing that arises from flat extrapolation
        of unknown future rates.

    The two-pass approach ensures that par bonds price within ~0.01 of par.

    Args:
        par_yields: Mapping of tenor (years) → par yield (decimal).
        as_of_date: Curve date (defaults to today).

    Returns:
        SpotCurve with interpolated spot rates.
    """
    if as_of_date is None:
        as_of_date = date.today()

    if not par_yields:
        raise ValueError("par_yields cannot be empty")

    tenors = sorted(par_yields.keys())
    coupon_freq = 2  # semi-annual

    logger.info(f"Bootstrapping spot curve with {len(tenors)} tenor points")

    # ── PASS 1: sequential bootstrap ──────────────────────────────────────────
    spot_rates_p1 = []
    for i, tenor in enumerate(tenors):
        par_yield = par_yields[tenor]

        if i == 0:
            # No previously known rates → flat-curve YTM equation
            z = _solve_spot_rate(tenor, par_yield, coupon_freq, interp_curve=None)
        else:
            # Build partial curve from already-bootstrapped tenors
            partial = SpotCurve(
                tenors=np.array(tenors[:i]),
                spot_rates=np.array(spot_rates_p1[:i]),
                as_of_date=as_of_date,
            )
            z = _solve_spot_rate(tenor, par_yield, coupon_freq, interp_curve=partial)

        spot_rates_p1.append(z)
        logger.debug(f"Pass 1 – tenor {tenor}Y: z = {z:.8f}")

    curve_p1 = SpotCurve(
        tenors=np.array(tenors),
        spot_rates=np.array(spot_rates_p1),
        as_of_date=as_of_date,
    )

    # ── PASS 2: refinement using full pass-1 curve ────────────────────────────
    # Re-solve each tenor using the complete pass-1 curve for intermediate coupons.
    # This makes the bootstrapped rates consistent with the final interpolated curve.
    spot_rates_p2 = []
    for i, tenor in enumerate(tenors):
        par_yield = par_yields[tenor]
        z = _solve_spot_rate(tenor, par_yield, coupon_freq, interp_curve=curve_p1)
        spot_rates_p2.append(z)
        logger.debug(f"Pass 2 – tenor {tenor}Y: z = {z:.8f}")

    curve = SpotCurve(
        tenors=np.array(tenors),
        spot_rates=np.array(spot_rates_p2),
        as_of_date=as_of_date,
    )

    # ── VALIDATION ────────────────────────────────────────────────────────────
    for i, tenor in enumerate(tenors):
        par_yield = par_yields[tenor]
        coupon_per_period = (par_yield / coupon_freq) * 100.0
        n_periods = int(round(tenor * coupon_freq))
        price = 0.0
        for k in range(1, n_periods + 1):
            t = k / coupon_freq
            if k < n_periods:
                price += coupon_per_period * curve.discount_factor(t)
            else:
                price += (coupon_per_period + 100.0) * curve.discount_factor(t)

        error = abs(price - 100.0)
        if error > 1e-2:
            logger.warning(f"Par bond validation: tenor {tenor}Y price = {price:.6f} (error: {error:.6f})")

    logger.info(f"Bootstrapped curve spans {tenors[0]}Y to {tenors[-1]}Y")
    return curve
