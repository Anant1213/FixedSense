"""Yield-to-Maturity (YTM) solver using Newton-Raphson optimization."""

import logging

import numpy as np
from scipy.optimize import brentq

from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


def solve_ytm(
    cashflows: CashFlowSchedule,
    market_price: float,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> float | None:
    """
    Solve for Yield-to-Maturity given bond cash flows and market price.

    Finds y such that:
    market_price = sum(CF_i / (1+y)^t_i)

    Uses Brentq root-finding (guaranteed convergence for continuous functions).

    Args:
        cashflows: Bond cash flow schedule
        market_price: Current bond price (as % of par)
        max_iterations: Maximum Newton-Raphson iterations (not used with Brentq)
        tolerance: Convergence tolerance

    Returns:
        YTM as decimal (e.g., 0.05 for 5%), or None if solver fails

    Raises:
        ValueError: If cash flows are invalid
    """
    if not cashflows.cash_flows:
        raise ValueError("No cash flows to discount")

    def bond_price(ytm: float) -> float:
        """Compute bond price as function of YTM."""
        pv = 0.0
        for cf in cashflows.cash_flows:
            pv += cf.amount / ((1.0 + ytm) ** cf.tenor)
        return pv

    def price_error(ytm: float) -> float:
        """Difference between computed and market price (to find zero)."""
        return bond_price(ytm) - market_price

    # Bracket for Brentq: ytm between -10% and 50%
    # Usually the bracket is much tighter, but we use wide bounds for safety
    ytm_min = -0.10
    ytm_max = 0.50

    try:
        ytm = brentq(price_error, ytm_min, ytm_max, xtol=tolerance)
        logger.debug(f"Solved YTM: {ytm:.6f} ({ytm*100:.4f}%)")
        return ytm

    except ValueError as e:
        logger.error(f"Failed to solve YTM: {e}")
        return None


def bond_price_from_ytm(
    cashflows: CashFlowSchedule,
    ytm: float,
) -> float:
    """
    Compute bond price from YTM.

    Args:
        cashflows: Bond cash flow schedule
        ytm: Yield to maturity as decimal

    Returns:
        Bond price as % of par (100 for par bond)
    """
    price = sum(cf.amount / ((1.0 + ytm) ** cf.tenor) for cf in cashflows.cash_flows)
    return price


def ytm_price_derivative(
    cashflows: CashFlowSchedule,
    ytm: float,
) -> float:
    """
    Compute analytical derivative: dPrice/dYTM.

    dPrice/dYTM = -sum(t_i * CF_i / (1+y)^(t_i+1))

    This is the negative of modified duration scaled by price.

    Args:
        cashflows: Bond cash flow schedule
        ytm: Yield to maturity

    Returns:
        Derivative (negative, since price decreases with yield)
    """
    derivative = 0.0
    for cf in cashflows.cash_flows:
        derivative -= cf.tenor * cf.amount / ((1.0 + ytm) ** (cf.tenor + 1))
    return derivative


def solve_ytm_newton_raphson(
    cashflows: CashFlowSchedule,
    market_price: float,
    initial_guess: float = 0.05,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> float | None:
    """
    Solve for YTM using Newton-Raphson method with analytical derivative.

    y_{n+1} = y_n - f(y_n) / f'(y_n)

    where f(y) = price(y) - market_price

    Note: Brentq (in solve_ytm) is preferred for robustness.

    Args:
        cashflows: Bond cash flow schedule
        market_price: Current bond price
        initial_guess: Starting ytm
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        YTM or None if fails
    """
    ytm = initial_guess

    for iteration in range(max_iterations):
        # Evaluate function and derivative
        price = bond_price_from_ytm(cashflows, ytm)
        f = price - market_price
        df = ytm_price_derivative(cashflows, ytm)

        # Check convergence
        if abs(f) < tolerance:
            logger.debug(f"Converged in {iteration+1} iterations")
            return ytm

        # Avoid division by zero
        if abs(df) < 1e-10:
            logger.warning(f"Derivative near zero at iteration {iteration}: {df}")
            return None

        # Newton-Raphson step
        ytm_new = ytm - f / df

        # Ensure we don't go too negative
        if ytm_new < -0.20:
            ytm_new = -0.20
        elif ytm_new > 1.0:
            ytm_new = 1.0

        ytm = ytm_new

    logger.warning(f"YTM solver did not converge after {max_iterations} iterations")
    return None


def ytm_spread(
    ytm_bond: float,
    ytm_benchmark: float,
) -> float:
    """
    Compute spread between bond YTM and benchmark (e.g., government) YTM.

    This is a simplified OAS (Option-Adjusted Spread).

    Args:
        ytm_bond: Bond YTM
        ytm_benchmark: Benchmark (e.g., G-sec) YTM

    Returns:
        Spread in basis points
    """
    spread = ytm_bond - ytm_benchmark
    return spread * 10000.0  # Convert to bps
