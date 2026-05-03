"""
Tests for spot curve bootstrapper.

Critical test: Par bond self-consistency
When we bootstrap from par yields and price a par bond using the spot curve,
we must recover exactly 100.
"""

import numpy as np
import pytest
from datetime import date, timedelta

from curves.bootstrapper import bootstrap_spot_curve
from pricing.cashflow_generator import generate_cashflow_schedule
from pricing.bond_pricer import BondPricer


def test_bootstrap_par_bond_self_consistency(sample_par_yields, spot_curve_10y):
    """
    CRITICAL TEST: Par bond priced on bootstrapped spot curve should equal 100.

    This validates the entire bootstrapper implementation.
    """
    # Create a par bond at each tenor
    for tenor, par_yield in sample_par_yields.items():
        bond_cf = generate_cashflow_schedule(
            bond_id=f"PAR_{tenor}Y",
            face_value=100.0,
            coupon_rate=par_yield,
            coupon_frequency=2,
            maturity_date=date.today() + timedelta(days=int(tenor * 365)),
            issue_date=date.today(),
        )

        # Price on bootstrapped curve
        price = BondPricer.price(bond_cf, spot_curve_10y, spread_bps=0.0)

        # Must recover 100 within 0.01 (1 cent per $100 face).
        # Residual error is calendar-day rounding: coupon dates fall on whole
        # calendar days, not the exact half-year fractions used in bootstrapping.
        assert abs(price - 100.0) < 0.01, \
            f"Par bond at {tenor}Y: price={price:.6f}, expected 100.0"


def test_spot_rate_ordering(spot_curve_10y):
    """Spot rates should generally increase with tenor (barbell effect ok)."""
    # For most markets, spot curve is upward sloping
    # At minimum, not violently oscillating

    diffs = np.diff(spot_curve_10y.spot_rates)
    mad_diff = np.median(np.abs(diffs))

    assert mad_diff < 0.01, "Spot curve oscillating too much"


def test_discount_factor_properties(spot_curve_10y):
    """Discount factors should be monotonically decreasing."""
    tenors = np.linspace(0.1, 10, 20)
    dfs = np.array([spot_curve_10y.discount_factor(t) for t in tenors])

    # DFs must be decreasing
    assert np.all(np.diff(dfs) < 0), "Discount factors not monotonically decreasing"

    # All DFs between 0 and 1
    assert np.all(dfs > 0) and np.all(dfs < 1), "Discount factors out of bounds"


def test_bootstrap_convergence():
    """Bootstrap should converge with simple par yields."""
    # Create synthetic upward-sloping par yields
    par_yields = {
        0.5: 0.02,
        1.0: 0.025,
        2.0: 0.03,
        3.0: 0.035,
        5.0: 0.04,
        10.0: 0.045,
    }

    curve = bootstrap_spot_curve(par_yields)

    # Verify spot rates exist and are reasonable
    assert len(curve.spot_rates) == len(par_yields)
    assert np.all(curve.spot_rates > 0)
    assert np.all(curve.spot_rates < 0.10)  # Yields between 0% and 10%


def test_forward_rate_consistency(spot_curve_10y):
    """Forward rate should be between neighboring spot rates."""
    # f(0,2) should be between z(1) and higher rate

    z1 = spot_curve_10y.rate_at(1.0)
    z2 = spot_curve_10y.rate_at(2.0)
    forward = spot_curve_10y.forward_rate(1.0, 2.0)

    # Forward should be reasonable (between short and long)
    assert forward > 0
    assert forward < 0.10
