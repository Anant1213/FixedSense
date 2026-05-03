"""Unit tests for factor attribution module."""

import numpy as np
import pytest
from datetime import date

from curves.bootstrapper import SpotCurve
from curves.pca_model import PCAModel
from pricing.cashflow_generator import CashFlowSchedule
from pnl.factor_attribution import FactorAttribution


@pytest.fixture
def simple_pca_model():
    """Create a simple PCA model for testing."""
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 0.5, 0.25, 0.083])
    loadings = np.random.randn(11, 3)  # 11 tenors, 3 factors
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


def test_factor_attribution_shape(sample_cashflows, simple_spot_curve, simple_pca_model):
    """Test that factor attribution returns correct shapes."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    exposures = FactorAttribution.compute_factor_exposures(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps, simple_pca_model
    )

    assert exposures.shape == (len(sample_cashflows), simple_pca_model.factor_loadings.shape[1])
    assert exposures.shape == (2, 3)


def test_zero_curve_change_zero_pnl(sample_cashflows, simple_spot_curve, simple_pca_model):
    """Test that zero curve change produces zero factor P&L."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]
    zero_curve_change = np.zeros(simple_pca_model.tenors.shape[0])

    result = FactorAttribution.compute(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, zero_curve_change
    )

    # Zero curve change should produce near-zero P&L
    assert np.allclose(result.total_factor_explained, 0.0, atol=1e-6)


def test_factor_pnl_sums(sample_cashflows, simple_spot_curve, simple_pca_model):
    """Test that factor P&L sums correctly."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    # Generate a reasonable curve change
    curve_change = np.random.randn(simple_pca_model.tenors.shape[0]) * 0.001

    result = FactorAttribution.compute(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps,
        simple_pca_model, curve_change
    )

    # Portfolio factor P&L should equal sum of per-bond factor P&L
    portfolio_total = result.total_factor_explained
    per_bond_sum = sum(
        bond_attr.total_factor_pnl for bond_attr in result.per_bond.values()
    )

    assert np.allclose(portfolio_total, per_bond_sum, atol=1e-3)


def test_empty_portfolio(simple_spot_curve, simple_pca_model):
    """Test that empty portfolio is handled gracefully."""
    result = FactorAttribution.compute(
        [], [], simple_spot_curve, [], simple_pca_model,
        np.zeros(simple_pca_model.tenors.shape[0])
    )

    assert len(result.per_bond) == 0
    assert result.total_factor_explained == 0.0


def test_factor_exposures_shape_consistency(sample_cashflows, simple_spot_curve, simple_pca_model):
    """Test that factor exposures shape is consistent."""
    notionals = [100e6, 150e6]
    spreads_bps = [0.0, 50.0]

    exp1 = FactorAttribution.compute_factor_exposures(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps, simple_pca_model
    )
    exp2 = FactorAttribution.compute_factor_exposures(
        sample_cashflows, notionals, simple_spot_curve, spreads_bps, simple_pca_model
    )

    # Same inputs should produce same outputs
    assert np.allclose(exp1, exp2)
