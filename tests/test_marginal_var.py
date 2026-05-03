"""Unit tests for marginal VaR decomposition."""

import numpy as np
import pytest
from datetime import date

from curves.bootstrapper import SpotCurve
from curves.pca_model import PCAModel
from pricing.cashflow_generator import CashFlowSchedule
from risk.monte_carlo import MonteCarloResult
from risk.marginal_var import MarginalVaRCalculator


@pytest.fixture
def simple_pca_model():
    """Create a simple PCA model for testing."""
    tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    loadings = np.random.randn(11, 3)
    eigenvalues = np.array([0.2, 0.1, 0.05])
    explained = eigenvalues / eigenvalues.sum()
    mean_changes = np.zeros(11)

    return PCAModel(
        factor_loadings=loadings,
        eigenvalues=eigenvalues,
        explained_variance=explained,
        mean_changes=mean_changes,
        tenors=tenors,
    )


@pytest.fixture
def simple_spot_curve():
    """Create a simple spot curve for testing."""
    return SpotCurve(
        tenors=np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
        spot_rates=np.array([0.04, 0.042, 0.044, 0.045, 0.048, 0.050, 0.052, 0.054, 0.055, 0.056, 0.057]),
        as_of_date=date(2026, 4, 30),
    )


@pytest.fixture
def sample_cashflows():
    """Create sample bond cash flows."""
    from pricing.cashflow_generator import generate_cashflow_schedule, DayCountMethod

    cf1 = generate_cashflow_schedule(
        bond_id="BOND1",
        face_value=100.0,
        coupon_rate=0.045,
        coupon_frequency=2,
        maturity_date=date(2031, 4, 30),
        issue_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 30),
        day_count_method=DayCountMethod.ACT_ACT,
    )

    cf2 = generate_cashflow_schedule(
        bond_id="BOND2",
        face_value=100.0,
        coupon_rate=0.050,
        coupon_frequency=2,
        maturity_date=date(2036, 4, 30),
        issue_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 30),
        day_count_method=DayCountMethod.ACT_ACT,
    )

    return [cf1, cf2]


@pytest.fixture
def sample_mc_result():
    """Create a sample Monte Carlo result."""
    n_scenarios = 1000
    n_factors = 3

    # Generate realistic P&L distribution
    factor_scores = np.random.multivariate_normal(
        np.zeros(n_factors),
        np.eye(n_factors),
        n_scenarios
    )

    pnl_dist = np.random.normal(0, 100, n_scenarios)  # $100 std dev

    return MonteCarloResult(
        pnl_distribution=pnl_dist,
        current_portfolio_value=500e6,
        shocked_portfolio_values=np.array([500e6 + p for p in pnl_dist]),
        factor_scores=factor_scores,
        curve_changes=np.random.randn(n_scenarios, 11),
    )


def test_var_decomposition_shapes(sample_cashflows, simple_spot_curve,
                                   simple_pca_model, sample_mc_result):
    """Test that VaR decomposition returns correct shapes."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    result = MarginalVaRCalculator.decompose(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, sample_mc_result
    )

    assert len(result.component_var) == len(sample_cashflows)
    assert len(result.marginal_var) == len(sample_cashflows)
    assert len(result.var_contribution_pct) == len(sample_cashflows)
    assert len(result.standalone_var) == len(sample_cashflows)


def test_component_var_sums_to_total(sample_cashflows, simple_spot_curve,
                                      simple_pca_model, sample_mc_result):
    """Test that component VaRs sum to total VaR (Euler property)."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    result = MarginalVaRCalculator.decompose(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, sample_mc_result, confidence=0.95
    )

    # Component VaR should sum to total VaR (Euler decomposition property)
    sum_component = sum(result.component_var.values())

    # Allow for numerical precision error
    assert np.allclose(sum_component, result.total_var, rtol=0.01, atol=1e-3)


def test_marginal_var_positive_for_long_bonds(sample_cashflows, simple_spot_curve,
                                               simple_pca_model, sample_mc_result):
    """Test that marginal VaR is positive for long bonds."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    result = MarginalVaRCalculator.decompose(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, sample_mc_result
    )

    # For long duration bonds, marginal VaR should generally be positive
    # (increases in portfolio notional increase VaR)
    for marginal_var in result.marginal_var.values():
        assert marginal_var >= -1e-6  # Allow for small numerical errors


def test_var_contribution_pcts_sum_to_one(sample_cashflows, simple_spot_curve,
                                           simple_pca_model, sample_mc_result):
    """Test that VaR contribution percentages sum to 100%."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    result = MarginalVaRCalculator.decompose(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, sample_mc_result
    )

    if result.total_var > 0:
        total_pct = sum(result.var_contribution_pct.values())
        assert np.allclose(total_pct, 1.0, rtol=0.01)


def test_diversification_benefit_calculation(sample_cashflows, simple_spot_curve,
                                             simple_pca_model, sample_mc_result):
    """Test that diversification benefit is computed correctly."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    result = MarginalVaRCalculator.decompose(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, sample_mc_result
    )

    # Diversification benefit = sum of standalone VaRs - portfolio VaR
    expected_benefit = sum(result.standalone_var.values()) - result.total_var
    assert np.allclose(result.diversification_benefit, expected_benefit, rtol=0.01, atol=1e-3)


def test_empty_portfolio(simple_spot_curve, simple_pca_model, sample_mc_result):
    """Test that empty portfolio is handled gracefully."""
    result = MarginalVaRCalculator.decompose(
        [], [], simple_spot_curve, [], simple_pca_model, sample_mc_result
    )

    assert len(result.component_var) == 0
    assert result.total_var >= 0  # Should be non-negative


def test_single_bond_var_equals_total(sample_cashflows, simple_spot_curve,
                                       simple_pca_model, sample_mc_result):
    """Test that single bond portfolio has component VaR equal to total VaR."""
    # Take only first bond
    cf_single = [sample_cashflows[0]]
    notionals_single = [100e6]
    spreads_single = [0.0]

    result = MarginalVaRCalculator.decompose(
        cf_single, notionals_single, simple_spot_curve, spreads_single,
        simple_pca_model, sample_mc_result
    )

    # For single bond, component VaR should equal total VaR
    assert np.allclose(
        result.component_var[cf_single[0].bond_id],
        result.total_var,
        rtol=0.1,
        atol=1e-3
    )
