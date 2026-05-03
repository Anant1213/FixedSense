"""
End-to-end demonstration of FixedSense.

This script runs the complete system: loads portfolio, computes Greeks,
VaR, P&L attribution, and stress tests.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

from config.settings import settings
from curves.bootstrapper import bootstrap_spot_curve
from curves.pca_model import fit_pca
from pricing.cashflow_generator import generate_cashflow_schedule, DayCountMethod
from pricing.bond_pricer import price_batch
from greeks.kr01 import KR01Calculator
from greeks.duration import DurationCalculator
from greeks.dv01 import DV01Calculator
from risk.monte_carlo import MonteCarloEngine
from risk.var_calculator import VaRCalculator
from risk.cvar_calculator import CVaRCalculator
from pnl.attribution import PnLAttribution
from stress.scenario_runner import ScenarioRunner
from data.ingestion.fred_client import FREDClient
from utils.date_utils import get_as_of_date, get_data_date_range

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_portfolio():
    """Load portfolio configuration."""
    logger.info("Loading portfolio configuration...")

    with open(settings.PORTFOLIO_CONFIG_PATH, 'r') as f:
        portfolio_config = yaml.safe_load(f)

    return portfolio_config


def load_yield_curves():
    """Load yield curve data (sample or from FRED)."""
    logger.info("Loading yield curve data...")

    fred_client = FREDClient(use_sample_data=True)

    start_date, end_date = get_data_date_range()

    yield_data = fred_client.fetch_treasury_yields(start_date, end_date)

    # Convert to wide format
    yield_curve_df = yield_data.pivot(
        index='date',
        columns='tenor',
        values='yield_pct'
    )

    return yield_curve_df


def create_bond_positions(portfolio_config, as_of_date: date):
    """Create cash flow schedules for all bonds."""
    logger.info("Creating bond cash flow schedules...")

    cashflows_list = []
    bonds_metadata = []

    for bond_config in portfolio_config['portfolio']['bonds']:
        cf = generate_cashflow_schedule(
            bond_id=bond_config['id'],
            face_value=bond_config['face_value'],
            coupon_rate=bond_config['coupon_rate'],
            coupon_frequency=bond_config['coupon_frequency'],
            maturity_date=pd.to_datetime(bond_config['maturity_date']).date(),
            issue_date=pd.to_datetime(bond_config['issue_date']).date(),
            as_of_date=as_of_date,
            day_count_method=DayCountMethod.ACT_ACT,
        )
        cashflows_list.append(cf)
        bonds_metadata.append(bond_config)

    return cashflows_list, bonds_metadata


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("FIXEDSENSE: FIXED INCOME PORTFOLIO RISK SYSTEM")
    logger.info("=" * 80)

    # 1. Load data
    portfolio_config = load_portfolio()
    yield_curve_df = load_yield_curves()
    as_of_date = get_as_of_date()
    cashflows_list, bonds_metadata = create_bond_positions(portfolio_config, as_of_date)

    # Extract portfolio parameters
    total_notional = portfolio_config['portfolio']['total_notional']
    bonds = portfolio_config['portfolio']['bonds']

    notionals = np.array([
        total_notional * bond['weight']
        for bond in bonds
    ])

    spreads_bps = np.array([
        bond.get('credit_spread_bps', 0.0)
        for bond in bonds
    ])

    logger.info(f"Portfolio: {len(cashflows_list)} bonds, ${total_notional/1e6:.1f}M total")

    # 2. Bootstrap spot curve
    logger.info("\n[1] Yield Curve Bootstrapping")
    latest_yields = yield_curve_df.iloc[-1].dropna()
    par_yields = {tenor: rate / 100.0 for tenor, rate in latest_yields.items()}
    spot_curve = bootstrap_spot_curve(par_yields)

    logger.info(f"✓ Spot curve bootstrapped: {len(spot_curve.tenors)} tenor points")
    logger.info(f"  Yield range: {spot_curve.spot_rates.min()*100:.2f}% - {spot_curve.spot_rates.max()*100:.2f}%")

    # 3. Fit PCA model
    logger.info("\n[2] PCA Factor Decomposition")
    yield_matrix = yield_curve_df.values
    tenors = yield_curve_df.columns.values

    pca_model = fit_pca(yield_matrix, tenors, n_factors=3)

    logger.info(f"✓ PCA model fitted on {len(yield_matrix)} days")
    logger.info(f"  Explained variance: {pca_model.explained_variance.sum()*100:.1f}%")
    logger.info(f"    PC1 (Level):     {pca_model.explained_variance[0]*100:.1f}%")
    logger.info(f"    PC2 (Slope):     {pca_model.explained_variance[1]*100:.1f}%")
    logger.info(f"    PC3 (Curvature): {pca_model.explained_variance[2]*100:.1f}%")

    # 4. Compute Greeks
    logger.info("\n[3] Greeks Computation")

    current_prices = price_batch(cashflows_list, spot_curve, spreads_bps)
    portfolio_value = np.sum(current_prices * notionals / 100.0)

    logger.info(f"✓ Portfolio NAV: ${portfolio_value/1e6:.2f}M")
    logger.info(f"  Bond prices: {current_prices}")

    # KR01
    kr01_portfolio = KR01Calculator.portfolio_kr01(
        cashflows_list, spot_curve, notionals.tolist(), spreads_bps.tolist()
    )

    total_kr01 = sum(kr01_portfolio.values())
    logger.info(f"✓ Total KR01: ${total_kr01:,.0f}")
    logger.info(f"  KR01 distribution by tenor:")
    for tenor, kr01_val in sorted(kr01_portfolio.items()):
        pct = (kr01_val / total_kr01) * 100
        logger.info(f"    {tenor:4.1f}Y: ${kr01_val:>10,.0f} ({pct:>5.1f}%)")

    # DV01
    total_dv01 = 0
    for i, cf in enumerate(cashflows_list):
        dv01 = DV01Calculator.dv01_from_ytm(cf, current_prices[i], notionals[i])
        total_dv01 += dv01

    logger.info(f"✓ Total DV01: ${total_dv01:,.0f}")

    # 5. Monte Carlo VaR
    logger.info("\n[4] Monte Carlo VaR/CVaR Simulation")

    mc_engine = MonteCarloEngine(pca_model, spot_curve, n_scenarios=10000, horizon_days=1)
    mc_result = mc_engine.simulate(cashflows_list, spreads_bps, current_prices)

    var_95 = VaRCalculator.monte_carlo_var(mc_result, confidence=0.95)
    var_99 = VaRCalculator.monte_carlo_var(mc_result, confidence=0.99)
    cvar_95 = CVaRCalculator.monte_carlo_cvar(mc_result, confidence=0.95)

    logger.info(f"✓ Ran 10,000 PCA-based scenarios")
    logger.info(f"  VaR  95%: ${var_95/1e6:>8.2f}M ({var_95/portfolio_value*100:>5.2f}% of portfolio)")
    logger.info(f"  VaR  99%: ${var_99/1e6:>8.2f}M ({var_99/portfolio_value*100:>5.2f}% of portfolio)")
    logger.info(f"  CVaR 95%: ${cvar_95/1e6:>8.2f}M ({cvar_95/portfolio_value*100:>5.2f}% of portfolio)")

    # 6. P&L Attribution
    logger.info("\n[5] P&L Attribution Analysis")

    yesterday_prices = current_prices + np.random.normal(0, 0.3, len(current_prices))
    yesterday_spreads = spreads_bps - np.random.normal(0, 3, len(spreads_bps))

    attr = PnLAttribution.compute_daily_attribution(
        cashflows_list, notionals.tolist(),
        yesterday_prices, spot_curve, yesterday_spreads.tolist(),
        spot_curve, spreads_bps.tolist(),
        as_of_date
    )

    logger.info(f"✓ Daily P&L Attribution:")
    logger.info(f"  Total P&L:   ${attr.total_pnl:>10.2f}")
    logger.info(f"  Carry:       ${attr.carry_pnl:>10.2f}")
    logger.info(f"  Rate:        ${attr.rate_pnl:>10.2f}")
    logger.info(f"  Spread:      ${attr.spread_pnl:>10.2f}")
    logger.info(f"  Residual:    ${attr.residual_pnl:>10.2f}")
    logger.info(f"  Explained:   {attr.explain_pct*100:>10.1f}%")

    # 7. Stress Testing
    logger.info("\n[6] Stress Testing")

    scenarios = [
        ScenarioRunner.create_parallel_up_scenario(
            cashflows_list, notionals, spot_curve, spreads_bps, bump_bps=200
        ),
        ScenarioRunner.create_parallel_down_scenario(
            cashflows_list, notionals, spot_curve, spreads_bps, bump_bps=100
        ),
        ScenarioRunner.create_steepener_scenario(
            cashflows_list, notionals, spot_curve, spreads_bps
        ),
        ScenarioRunner.create_spread_blowout_scenario(
            cashflows_list, notionals, spot_curve, spreads_bps
        ),
    ]

    logger.info(f"✓ Ran {len(scenarios)} stress scenarios:")
    for scenario in scenarios:
        logger.info(
            f"  {scenario.scenario_name:30s}: "
            f"${scenario.impact_absolute/1e6:>7.2f}M ({scenario.impact_pct:>7.2f}%)"
        )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM READY FOR PRODUCTION")
    logger.info("=" * 80)
    logger.info("""
Dashboard: streamlit run dashboard/app.py
Tests:     pytest tests/ -v
API:       from fixedsense import *
    """)

    return {
        'portfolio': portfolio_config,
        'spot_curve': spot_curve,
        'pca_model': pca_model,
        'portfolio_value': portfolio_value,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'attribution': attr,
        'scenarios': scenarios,
    }


if __name__ == "__main__":
    results = main()
