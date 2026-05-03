"""
Tests for P&L attribution.

CRITICAL: Attribution identity must hold.
carry + rate + spread + residual = total_pnl (within Rs 1)
"""

import numpy as np
import pytest
from datetime import date, timedelta

from curves.bootstrapper import bootstrap_spot_curve
from pricing.cashflow_generator import generate_cashflow_schedule
from pricing.bond_pricer import price_batch
from pnl.attribution import PnLAttribution


def test_attribution_identity_holds():
    """
    CRITICAL: Components sum to total within tolerance.

    carry + rate + spread + residual = total_pnl (within Rs 1)
    """
    # Setup
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    yesterday_curve = bootstrap_spot_curve(par_yields, as_of_date=date.today() - timedelta(days=1))
    today_curve = bootstrap_spot_curve(par_yields, as_of_date=date.today())

    # Create test bonds
    cashflows_list = []
    for i, tenor in enumerate([3, 5, 7]):
        bond_cf = generate_cashflow_schedule(
            bond_id=f"TEST_{tenor}Y",
            face_value=100.0,
            coupon_rate=0.035 + i * 0.005,
            coupon_frequency=2,
            maturity_date=date.today() + timedelta(days=365 * tenor),
            issue_date=date.today(),
        )
        cashflows_list.append(bond_cf)

    notionals = np.array([1e7, 1.5e7, 2e7])
    yesterday_spreads = np.array([0.0, 50.0, 75.0])
    today_spreads = np.array([0.0, 55.0, 80.0])

    yesterday_prices = price_batch(cashflows_list, yesterday_curve, yesterday_spreads)

    # Compute attribution
    attr = PnLAttribution.compute_daily_attribution(
        cashflows_list, notionals.tolist(),
        yesterday_prices, yesterday_curve, yesterday_spreads.tolist(),
        today_curve, today_spreads.tolist(),
        date.today()
    )

    # CRITICAL: Check identity
    computed_sum = attr.carry_pnl + attr.rate_pnl + attr.spread_pnl + attr.residual_pnl
    error = abs(computed_sum - attr.total_pnl)

    assert error < 1.0, \
        f"Attribution identity failed: {computed_sum:.2f} + residual != {attr.total_pnl:.2f}, error={error:.2f}"


def test_attribution_carry_always_positive():
    """Carry P&L should always be positive (you earn coupon)."""
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    curve = bootstrap_spot_curve(par_yields)

    cashflows_list = []
    for tenor in [3, 5]:
        bond_cf = generate_cashflow_schedule(
            bond_id=f"TEST_{tenor}Y",
            face_value=100.0,
            coupon_rate=0.04,
            coupon_frequency=2,
            maturity_date=date.today() + timedelta(days=365 * tenor),
            issue_date=date.today(),
        )
        cashflows_list.append(bond_cf)

    notionals = np.array([1e7, 1.5e7])
    spreads = np.array([0.0, 50.0])

    yesterday_prices = price_batch(cashflows_list, curve, spreads)

    attr = PnLAttribution.compute_daily_attribution(
        cashflows_list, notionals.tolist(),
        yesterday_prices, curve, spreads.tolist(),
        curve, spreads.tolist(),
        date.today()
    )

    assert attr.carry_pnl > 0, f"Carry should be positive, got {attr.carry_pnl:.2f}"


def test_attribution_validation():
    """Test the validation function."""
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    curve = bootstrap_spot_curve(par_yields)

    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365 * 5),
        issue_date=date.today(),
    )

    notionals = np.array([1e7])
    spreads = np.array([0.0])

    yesterday_prices = price_batch([bond_cf], curve, spreads)

    attr = PnLAttribution.compute_daily_attribution(
        [bond_cf], notionals.tolist(),
        yesterday_prices, curve, spreads.tolist(),
        curve, spreads.tolist(),
        date.today()
    )

    # Should validate successfully
    is_valid = PnLAttribution.validate_attribution(attr, tolerance=1.0)
    assert is_valid, "Valid attribution failed validation"


def test_per_bond_attribution_sums():
    """Per-bond attribution components should sum to total."""
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    yesterday_curve = bootstrap_spot_curve(par_yields)
    today_curve = bootstrap_spot_curve(par_yields)

    bond_cf = generate_cashflow_schedule(
        bond_id="TEST_5Y",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365 * 5),
        issue_date=date.today(),
    )

    notionals = np.array([1e7])
    spreads = np.array([0.0])

    yesterday_prices = price_batch([bond_cf], yesterday_curve, spreads)

    attr = PnLAttribution.compute_daily_attribution(
        [bond_cf], notionals.tolist(),
        yesterday_prices, yesterday_curve, spreads.tolist(),
        today_curve, spreads.tolist(),
        date.today()
    )

    # Check per-bond attribution
    bond_attr = attr.per_bond["TEST_5Y"]

    computed_bond_sum = (
        bond_attr.carry_pnl + bond_attr.rate_pnl +
        bond_attr.spread_pnl + bond_attr.residual_pnl
    )

    error = abs(computed_bond_sum - bond_attr.total_pnl)

    assert error < 0.01, f"Per-bond attribution doesn't sum: error={error:.4f}"
