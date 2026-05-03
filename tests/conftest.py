"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from curves.bootstrapper import bootstrap_spot_curve
from curves.pca_model import fit_pca
from pricing.cashflow_generator import generate_cashflow_schedule, CashFlowSchedule
from config.settings import settings


@pytest.fixture
def sample_par_yields():
    """Sample par yields for testing."""
    return {
        1.0: 0.03,
        2.0: 0.035,
        3.0: 0.037,
        5.0: 0.04,
        7.0: 0.042,
        10.0: 0.045,
    }


@pytest.fixture
def spot_curve_10y(sample_par_yields):
    """10-year spot curve from par yields."""
    return bootstrap_spot_curve(sample_par_yields, as_of_date=date.today())


@pytest.fixture
def sample_bond():
    """Sample 5Y corporate bond."""
    return generate_cashflow_schedule(
        bond_id="TEST_5Y",
        face_value=100.0,
        coupon_rate=0.045,
        coupon_frequency=2,
        maturity_date=date.today() + timedelta(days=365*5),
        issue_date=date.today() - timedelta(days=365),
    )


@pytest.fixture
def historical_yields():
    """Generate synthetic historical yields for PCA testing."""
    np.random.seed(42)

    dates = pd.date_range(end=date.today(), periods=504, freq='B')
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    # Synthetic yields with realistic patterns
    base_yields = np.array([0.03, 0.035, 0.037, 0.04, 0.042, 0.045])
    yields = np.zeros((len(dates), len(tenors)))

    for t in range(len(dates)):
        # Add realistic movements (level, slope, curvature changes)
        level_change = np.random.normal(0, 0.001)
        slope_change = np.random.normal(0, 0.0005)
        curve_change = np.random.normal(0, 0.0003)

        yields[t, :] = base_yields + level_change + slope_change * (tenors - tenors.mean()) + curve_change * ((tenors - tenors.mean())**2)

    df = pd.DataFrame(yields, columns=tenors, index=dates)
    return df


@pytest.fixture
def pca_model(historical_yields):
    """Fitted PCA model."""
    return fit_pca(historical_yields.values, historical_yields.columns.values, n_factors=3)
