"""
Tests for Greeks: duration, DV01, KR01, convexity.

Critical properties:
1. Zero-coupon bond: duration equals maturity
2. KR01 sum ≈ total DV01 (within 5%)
3. Convexity always positive
"""

import numpy as np
import pytest
from datetime import date, timedelta

from curves.bootstrapper import bootstrap_spot_curve
from pricing.cashflow_generator import generate_cashflow_schedule, DayCountMethod
from pricing.ytm_solver import solve_ytm
from pricing.bond_pricer import BondPricer
from greeks.duration import DurationCalculator
from greeks.dv01 import DV01Calculator
from greeks.kr01 import KR01Calculator
from greeks.convexity import ConvexityCalculator


def test_zero_coupon_bond_duration():
    """
    CRITICAL: Zero-coupon bond duration = maturity.

    A zero-coupon bond maturing in T years has Macaulay duration = T.
    """
    # Create 5Y zero-coupon bond
    maturity_date = date.today() + timedelta(days=365*5)
    bond_cf = generate_cashflow_schedule(
        bond_id="ZERO_5Y",
        face_value=100.0,
        coupon_rate=0.0,
        coupon_frequency=2,
        maturity_date=maturity_date,
        issue_date=date.today(),
    )

    ytm = 0.03  # Assume 3% YTM

    macaulay_dur = DurationCalculator.macaulay_duration(bond_cf, ytm)

    # Must equal 5 years (within 0.01 year tolerance)
    assert abs(macaulay_dur - 5.0) < 0.01, \
        f"Zero-coupon bond duration={macaulay_dur:.4f}, expected 5.0"


def test_par_bond_duration_less_than_maturity():
    """Par bond duration < maturity (because of coupons)."""
    maturity_years = 5
    maturity_date = date.today() + timedelta(days=int(maturity_years * 365))

    bond_cf = generate_cashflow_schedule(
        bond_id="PAR_5Y",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=maturity_date,
        issue_date=date.today(),
    )

    ytm = 0.04  # Par bond: YTM = coupon rate

    macaulay_dur = DurationCalculator.macaulay_duration(bond_cf, ytm)

    # Duration must be < maturity
    assert macaulay_dur < maturity_years, \
        f"Duration {macaulay_dur:.2f}Y should be < maturity {maturity_years}Y"

    # For par bond with reasonable coupons, duration is 60-80% of maturity
    assert macaulay_dur > maturity_years * 0.5, \
        f"Duration {macaulay_dur:.2f}Y seems too short"


def test_kr01_sum_approximates_dv01():
    """
    CRITICAL: Sum of KR01 across all tenors ≈ total DV01.

    Within 5% tolerance (some difference due to interpolation).
    """
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    spot_curve = bootstrap_spot_curve(par_yields)

    bond_cf = generate_cashflow_schedule(
        bond_id="TEST_5Y",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    notional = 1e7  # Rs 1 Cr

    # Compute KR01
    kr01_result = KR01Calculator.compute_kr01(bond_cf, spot_curve, notional, spread_bps=0.0)
    total_kr01 = kr01_result.total_kr01

    # Compute DV01
    bond_price = BondPricer.price(bond_cf, spot_curve, spread_bps=0.0)
    ytm = solve_ytm(bond_cf, bond_price)
    assert ytm is not None

    mod_dur = DurationCalculator.modified_duration(bond_cf, ytm)
    dv01 = mod_dur * (bond_price / 100.0) * (notional / 10000.0)

    # KR01 sum should be within 5% of DV01
    relative_error = abs(total_kr01 - dv01) / dv01

    assert relative_error < 0.05, \
        f"KR01 sum ({total_kr01:.2f}) vs DV01 ({dv01:.2f}): {relative_error*100:.2f}% error"


def test_convexity_always_positive():
    """
    CRITICAL: Vanilla bond convexity is always positive.

    Negative convexity indicates an error (likely sign error in formula).
    """
    bond_cf = generate_cashflow_schedule(
        bond_id="TEST_10Y",
        face_value=100.0,
        coupon_rate=0.045,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*10),
        issue_date=date.today(),
    )

    ytm = 0.045

    convexity = ConvexityCalculator.convexity_from_ytm(bond_cf, ytm)

    assert convexity > 0, f"Convexity {convexity:.4f} must be positive"


def test_dv01_proportional_to_notional():
    """DV01 scales linearly with notional."""
    par_yields = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
    spot_curve = bootstrap_spot_curve(par_yields)

    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    bond_price = BondPricer.price(bond_cf, spot_curve)
    ytm = solve_ytm(bond_cf, bond_price)
    assert ytm is not None

    # DV01 at 1Cr notional
    dv01_1cr = DV01Calculator.dv01_from_ytm(bond_cf, bond_price, 1e7)

    # DV01 at 2Cr notional
    dv01_2cr = DV01Calculator.dv01_from_ytm(bond_cf, bond_price, 2e7)

    # Must be exactly 2x
    assert abs(dv01_2cr - 2 * dv01_1cr) < 1.0, \
        f"DV01 not scaling linearly with notional"


def test_modified_duration_positive():
    """Modified duration is always positive for vanilla bonds."""
    bond_cf = generate_cashflow_schedule(
        bond_id="TEST",
        face_value=100.0,
        coupon_rate=0.04,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today(),
    )

    ytm = 0.03

    mod_dur = DurationCalculator.modified_duration(bond_cf, ytm)

    assert mod_dur > 0, f"Modified duration {mod_dur:.4f} must be positive"
