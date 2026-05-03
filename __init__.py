"""
FixedSense: Production-Grade Fixed Income Risk Management System

High-level API for common operations.
"""

from config.settings import settings
from curves.bootstrapper import bootstrap_spot_curve, SpotCurve
from curves.pca_model import fit_pca, PCAModel
from pricing.bond_pricer import BondPricer
from pricing.cashflow_generator import generate_cashflow_schedule
from greeks.kr01 import KR01Calculator
from risk.monte_carlo import MonteCarloEngine, MonteCarloResult
from risk.var_calculator import VaRCalculator
from risk.cvar_calculator import CVaRCalculator

__version__ = "1.0.0"
__author__ = "Anant Srivastava"

__all__ = [
    "settings",
    "bootstrap_spot_curve",
    "SpotCurve",
    "fit_pca",
    "PCAModel",
    "BondPricer",
    "generate_cashflow_schedule",
    "KR01Calculator",
    "MonteCarloEngine",
    "MonteCarloResult",
    "VaRCalculator",
    "CVaRCalculator",
]
