"""Implied forward rate computation from spot curves."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def forward_rate(
    z_t1: float,
    z_t2: float,
    t1: float,
    t2: float,
) -> float:
    """
    Compute implied forward rate between two tenors.

    Formula:
    f(t1,t2) = ((1+z(t2))^t2 / (1+z(t1))^t1)^(1/(t2-t1)) - 1

    This is the rate implied by the current spot curve for borrowing
    from time t1 to time t2.

    Args:
        z_t1: Spot rate at t1 (decimal)
        z_t2: Spot rate at t2 (decimal)
        t1: First tenor (years)
        t2: Second tenor (years, must be > t1)

    Returns:
        Forward rate as decimal

    Raises:
        ValueError: If t2 <= t1
    """
    if t2 <= t1:
        raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")

    # (1 + z(t2))^t2 / (1 + z(t1))^t1
    ratio = ((1.0 + z_t2) ** t2) / ((1.0 + z_t1) ** t1)

    # Take the (1/(t2-t1))-th power and subtract 1
    forward = ratio ** (1.0 / (t2 - t1)) - 1.0

    return forward


def forward_rates_vector(
    spot_curve_data: dict[float, float],
    forward_start: float,
) -> dict[float, float]:
    """
    Compute 1-year forward rates starting at various points.

    E.g., forward_rates_vector({1: 0.03, 2: 0.035, 3: 0.037}, forward_start=1)
    returns forward rates f(0,1), f(1,2), f(2,3), ...

    Args:
        spot_curve_data: Dict mapping tenor (years) to spot rate (decimal)
        forward_start: Duration of forward period (e.g., 1 for 1-year forwards)

    Returns:
        Dict mapping tenor to forward rate starting at that tenor
    """
    tenors = sorted(spot_curve_data.keys())
    forwards = {}

    for i in range(len(tenors) - 1):
        t1 = tenors[i]
        t2 = t1 + forward_start

        # Find spot rates at t1 and t2
        # If t2 not in data, interpolate (simplified: use next available)
        if t2 in spot_curve_data:
            z_t2 = spot_curve_data[t2]
        else:
            # Find the next tenor >= t2
            next_tenors = [t for t in tenors if t > t1]
            if not next_tenors:
                break
            t2 = next_tenors[0]
            z_t2 = spot_curve_data[t2]

        z_t1 = spot_curve_data[t1]

        f = forward_rate(z_t1, z_t2, t1, t2)
        forwards[t1] = f

    return forwards


def par_yield_from_spot_curve(
    spot_rates: dict[float, float],
    tenor: float,
    coupon_freq: int = 2,
) -> float:
    """
    Compute par bond yield from spot curve.

    The par yield is the coupon rate at which a bond with given tenor
    trades at par (100).

    Using spot rates, the price of a bond with coupon c and tenor T is:
    P = sum(c/(1+z(t))^t for t in coupons) + 100/(1+z(T))^T

    Setting P = 100 and solving for c gives the par yield.

    Args:
        spot_rates: Dict mapping tenor (years) to spot rate
        tenor: Bond maturity (years)
        coupon_freq: Coupon frequency (2 for semi-annual, 1 for annual)

    Returns:
        Par yield as decimal
    """
    # We need to interpolate spot rates for all coupon dates
    # Simplified: use adjacent tenors from the dict
    tenors = sorted(spot_rates.keys())

    if tenor not in tenors:
        # Interpolate: find surrounding tenors
        lower = [t for t in tenors if t <= tenor]
        upper = [t for t in tenors if t > tenor]

        if not lower or not upper:
            # Can't interpolate, use nearest
            nearest = min(tenors, key=lambda t: abs(t - tenor))
            return spot_rates[nearest]

        # Linear interpolation of spot rates
        t_low, t_up = lower[-1], upper[0]
        z_low, z_up = spot_rates[t_low], spot_rates[t_up]
        z_tenor = z_low + (z_up - z_low) * (tenor - t_low) / (t_up - t_low)
    else:
        z_tenor = spot_rates[tenor]

    # For a par bond: P = 100
    # 100 = sum(c/freq * 1/(1+z(t))^t) + 100/(1+z(T))^T
    # Rearrange to solve for c

    pv_principal = 100.0 / ((1.0 + z_tenor) ** tenor)

    # PV of annuity of coupons: need to sum over coupon dates
    # This is an approximation using Macaulay duration formula
    # For accuracy, we'd need to interpolate spot rates at each coupon date

    # Simplified calculation using duration
    # Par yield ≈ spot rate at tenor for short tenors
    # More precise: iterative or closed-form annuity formula

    # For now, use simplified annuity PV
    pv_annuity = 0.0
    for period in range(1, int(tenor * coupon_freq) + 1):
        t = period / coupon_freq
        # Approximate spot rate at this time
        if t in spot_rates:
            z_t = spot_rates[t]
        else:
            # Linear interpolation
            tenors = sorted(spot_rates.keys())
            lower = [tt for tt in tenors if tt <= t]
            upper = [tt for tt in tenors if tt > t]
            if lower and upper:
                t_low, t_up = lower[-1], upper[0]
                z_low, z_up = spot_rates[t_low], spot_rates[t_up]
                z_t = z_low + (z_up - z_low) * (t - t_low) / (t_up - t_low)
            else:
                z_t = z_tenor  # Fallback

        pv_annuity += 1.0 / ((1.0 + z_t) ** t)

    # Par yield: solve 100 = c/freq * pv_annuity + 100 * pv_principal
    # c = freq * (100 - 100*pv_principal) / pv_annuity
    coupon = coupon_freq * (100.0 - 100.0 * pv_principal) / pv_annuity

    return coupon / 100.0  # Return as decimal (yield)
