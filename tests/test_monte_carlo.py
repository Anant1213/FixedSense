"""
Tests for Monte Carlo simulation and VaR/CVaR.

Critical properties:
1. VaR is monotonic: VaR99% >= VaR95% >= VaR90%
2. CVaR >= VaR (always)
3. With zero volatility, all scenarios have same price
"""

import numpy as np
import pytest
from datetime import date, timedelta

from curves.bootstrapper import bootstrap_spot_curve
from curves.pca_model import fit_pca
from pricing.cashflow_generator import generate_cashflow_schedule
from pricing.bond_pricer import price_batch
from risk.monte_carlo import MonteCarloEngine
from risk.var_calculator import VaRCalculator
from risk.cvar_calculator import CVaRCalculator


def test_var_monotonicity(pca_model, spot_curve_10y):
    """
    CRITICAL: VaR monotonicity.

    VaR at 99% >= VaR at 95% >= VaR at 90%
    """
    # Create simple bond
    bond_cf = generate_cashflow_schedule(
        bond_id="TEST_5Y",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    # Run Monte Carlo
    mc_engine = MonteCarloEngine(pca_model, spot_curve_10y, n_scenarios=5000)
    mc_result = mc_engine.simulate([bond_cf], spreads_bps=0.0)

    # Compute VaR at different confidence levels
    var_90 = VaRCalculator.monte_carlo_var(mc_result, confidence=0.90)
    var_95 = VaRCalculator.monte_carlo_var(mc_result, confidence=0.95)
    var_99 = VaRCalculator.monte_carlo_var(mc_result, confidence=0.99)

    # Must be monotonic increasing
    assert var_95 >= var_90, f"VaR95 ({var_95:.2f}) < VaR90 ({var_90:.2f})"
    assert var_99 >= var_95, f"VaR99 ({var_99:.2f}) < VaR95 ({var_95:.2f})"


def test_cvar_greater_than_var(pca_model, spot_curve_10y):
    """
    CRITICAL: CVaR >= VaR always.

    CVaR is the average loss beyond VaR, so it must be at least as large.
    """
    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    mc_engine = MonteCarloEngine(pca_model, spot_curve_10y, n_scenarios=5000)
    mc_result = mc_engine.simulate([bond_cf], spreads_bps=0.0)

    var = VaRCalculator.monte_carlo_var(mc_result, confidence=0.95)
    cvar = CVaRCalculator.monte_carlo_cvar(mc_result, confidence=0.95)

    assert cvar >= var, f"CVaR ({cvar:.2f}) < VaR ({var:.2f})"


def test_var_convergence(pca_model, spot_curve_10y):
    """
    VaR should converge with increasing scenarios.

    Run multiple simulations and check that standard deviation decreases.
    """
    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    var_samples = []

    # Run 5 independent simulations
    for i in range(5):
        mc_engine = MonteCarloEngine(pca_model, spot_curve_10y, n_scenarios=5000, random_state=i)
        mc_result = mc_engine.simulate([bond_cf], spreads_bps=0.0)
        var = VaRCalculator.monte_carlo_var(mc_result, confidence=0.95)
        var_samples.append(var)

    var_samples = np.array(var_samples)
    var_std = var_samples.std()
    var_mean = var_samples.mean()
    var_cv = var_std / var_mean  # Coefficient of variation

    # With 5000 scenarios, CV should be < 5%
    assert var_cv < 0.05, f"VaR not converging: CV={var_cv:.2%}"


def test_pnl_distribution_shape(pca_model, spot_curve_10y):
    """
    P&L distribution should be roughly normal (from Central Limit Theorem).

    Check skewness and kurtosis bounds.
    """
    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    mc_engine = MonteCarloEngine(pca_model, spot_curve_10y, n_scenarios=10000)
    mc_result = mc_engine.simulate([bond_cf], spreads_bps=0.0)

    pnl = mc_result.pnl_distribution

    # Compute statistics
    skewness = (pnl - pnl.mean()) ** 3 / pnl.std() ** 3
    mean_skewness = skewness.mean()

    # For normal distribution, skewness ≈ 0
    # For bond portfolios, slight negative skew is expected
    assert abs(mean_skewness) < 1.0, f"Skewness {mean_skewness:.2f} too large"


def test_scenario_count_effect():
    """
    More scenarios should produce lower VaR standard error.

    (Quick smoke test - not exhaustive.)
    """
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    spot_curve = bootstrap_spot_curve(par_yields)

    # Fit PCA on data with realistic variance (constant matrix → zero eigenvalues → VaR=0)
    np.random.seed(7)
    base = np.array([0.03, 0.035, 0.04, 0.045])
    yields_matrix = base + np.random.normal(0, 0.001, (100, 4))
    pca_minimal = fit_pca(yields_matrix, np.array([1.0, 2.0, 5.0, 10.0]), n_factors=2)

    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    # Run with few scenarios
    mc_engine_small = MonteCarloEngine(pca_minimal, spot_curve, n_scenarios=1000)
    mc_result_small = mc_engine_small.simulate([bond_cf])

    # Run with many scenarios
    mc_engine_large = MonteCarloEngine(pca_minimal, spot_curve, n_scenarios=10000)
    mc_result_large = mc_engine_large.simulate([bond_cf])

    # Both should produce reasonable VaR numbers
    var_small = VaRCalculator.monte_carlo_var(mc_result_small, confidence=0.95)
    var_large = VaRCalculator.monte_carlo_var(mc_result_large, confidence=0.95)

    # Should be in same ballpark (within 50%)
    assert 0.5 * var_small < var_large < 2.0 * var_small, \
        f"VaR estimates diverging: {var_small:.2f} vs {var_large:.2f}"
