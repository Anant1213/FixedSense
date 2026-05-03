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
        Get interpolated spot rate at any tenor.

        Args:
            tenor: Tenor in years

        Returns:
            Spot rate as decimal (0.05 for 5%)
        """
        if tenor < self.tenors[0]:
            # Extrapolate flat to left
            return self.spot_rates[0]
        elif tenor > self.tenors[-1]:
            # Extrapolate flat to right
            return self.spot_rates[-1]
        else:
            # Interpolate using cubic spline
            cs = CubicSpline(self.tenors, self.spot_rates)
            return float(cs(tenor))

    def discount_factor(self, tenor: float) -> float:
        """
        Get discount factor at tenor.

        DF = 1 / (1 + z(t))^t

        Args:
            tenor: Tenor in years

        Returns:
            Discount factor (0 to 1)
        """
        z = self.rate_at(tenor)
        return 1.0 / ((1.0 + z) ** tenor)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Implied forward rate between two tenors.

        f(t1,t2) = ((1+z(t2))^t2 / (1+z(t1))^t1)^(1/(t2-t1)) - 1

        Args:
            t1: Start tenor (years)
            t2: End tenor (years, must be > t1)

        Returns:
            Forward rate as decimal
        """
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be > t1 ({t1})")

        z1 = self.rate_at(t1)
        z2 = self.rate_at(t2)

        ratio = ((1.0 + z2) ** t2) / ((1.0 + z1) ** t1)
        forward = ratio ** (1.0 / (t2 - t1)) - 1.0

        return forward


def bootstrap_spot_curve(
    par_yields: dict[float, float],
    as_of_date: date = None,
) -> SpotCurve:
    """
    Bootstrap zero-coupon spot curve from par bond yields.

    Algorithm:
    1. For the shortest tenor t1: spot rate = par yield (par bond has zero accrual)
    2. For each subsequent tenor ti:
       Set up equation: Par Bond Price = 100
       100 = sum(coupon / (1 + z(tj))^tj for j < i) + (100 + coupon) / (1 + z(ti))^ti
       All z(tj) for j < i are known.
       Solve for z(ti) using Brentq root-finding.

    Args:
        par_yields: Dictionary mapping tenor (years) to par yield (decimal).
                    E.g., {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, ...}
        as_of_date: Date for which this curve was computed

    Returns:
        SpotCurve with interpolated spot rates

    Raises:
        ValueError: If input data is invalid
    """
    if as_of_date is None:
        as_of_date = date.today()

    if not par_yields:
        raise ValueError("par_yields cannot be empty")

    # Sort tenors
    tenors = sorted(par_yields.keys())
    spot_rates = []

    logger.info(f"Bootstrapping spot curve with {len(tenors)} tenor points")

    # Semi-annual coupon payment (typical for bonds)
    coupon_freq = 2

    for i, tenor in enumerate(tenors):
        par_yield = par_yields[tenor]

        if i == 0:
            # First tenor: spot rate = par rate (par bond has zero accrued interest)
            spot_rate = par_yield
            logger.debug(f"Tenor {tenor}Y: spot_rate = {spot_rate:.6f} (par)")

        else:
            # Subsequent tenors: solve for spot rate using Brentq
            # Capture loop variables by value for the closure
            _i = i
            _tenor = tenor
            _par_yield = par_yield
            _known_tenors = list(tenors[:i])
            _known_rates = list(spot_rates[:i])

            def par_bond_price(z: float) -> float:
                """
                Par bond price as function of the unknown spot rate z at _tenor.

                Correct algorithm:
                - Dollar coupon per period = (par_yield / freq) × 100  (for $100 face)
                - For each semi-annual coupon period k = 1 .. n_periods:
                  - If k < n_periods: discount coupon at INTERPOLATED known spot rate
                  - If k == n_periods (final): discount coupon + $100 principal at z
                """
                dollar_coupon = (_par_yield / coupon_freq) * 100.0  # e.g. 1.75 for 3.5% semi
                n_periods = int(round(_tenor * coupon_freq))
                pv = 0.0

                for k in range(1, n_periods + 1):
                    t = k / coupon_freq
                    if k < n_periods:
                        # Intermediate coupon: use interpolated known spot rates
                        rate = np.interp(t, _known_tenors, _known_rates)
                        pv += dollar_coupon / ((1.0 + rate) ** t)
                    else:
                        # Final: coupon + $100 principal at unknown z
                        pv += (dollar_coupon + 100.0) / ((1.0 + z) ** t)

                return pv - 100.0

            # Bracket search: spot rate between 0.0001 and 0.50
            try:
                spot_rate = brentq(par_bond_price, 0.0001, 0.50, xtol=1e-8)
                logger.debug(f"Tenor {tenor}Y: spot_rate = {spot_rate:.6f}")
            except ValueError as e:
                logger.error(f"Failed to bootstrap tenor {tenor}Y: {e}")
                raise ValueError(f"Cannot bootstrap tenor {tenor}Y") from e

        spot_rates.append(spot_rate)

    # Create curve
    curve = SpotCurve(
        tenors=np.array(tenors),
        spot_rates=np.array(spot_rates),
        as_of_date=as_of_date,
    )

    # Validate: price a par bond on the spot curve, should recover 100
    for i, tenor in enumerate(tenors):
        par_yield = par_yields[tenor]
        coupon_per_period = par_yield / coupon_freq
        price = 0.0

        for period in range(1, int(tenor * coupon_freq) + 1):
            time = period / coupon_freq
            df = curve.discount_factor(time)
            price += coupon_per_period * df

        price += 100.0 * curve.discount_factor(tenor)

        error = abs(price - 100.0)
        if error > 1e-4:
            logger.warning(f"Par bond validation: tenor {tenor}Y price = {price:.6f} (error: {error:.6f})")

    logger.info(f"Bootstrapped curve spans {tenors[0]}Y to {tenors[-1]}Y")
    return curve
