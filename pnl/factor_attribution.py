"""
PCA factor attribution: decompose P&L and risk exposures into level, slope, curvature.

This is what quants use to understand which macro drivers are driving portfolio P&L.
Key formula: factor_exposure[bond, factor] = KR01[bond] @ PCA_loadings[:, factor]
Then: factor_pnl[bond, factor] = factor_exposure[bond, factor] × realized_factor_score[factor]
"""

import logging
from dataclasses import dataclass

import numpy as np

from config.settings import settings
from curves.pca_model import PCAModel
from greeks.kr01 import KR01Calculator
from pricing.cashflow_generator import CashFlowSchedule
from curves.bootstrapper import SpotCurve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BondFactorAttribution:
    """Factor attribution for a single bond."""

    bond_id: str
    factor_pnl: dict[str, float]  # "PC1 (Level)" → pnl
    total_factor_pnl: float
    actual_pnl: float
    residual_pnl: float
    explained_pct: float


@dataclass(frozen=True)
class FactorAttributionResult:
    """Full portfolio factor attribution result."""

    portfolio_factor_pnl: dict[str, float]  # PC1, PC2, PC3 contributions
    per_bond: dict[str, BondFactorAttribution]
    total_factor_explained: float
    actual_portfolio_pnl: float
    residual_pnl: float
    factor_exposures: np.ndarray  # shape (n_bonds, n_factors)
    realized_factor_scores: np.ndarray  # shape (n_factors,)
    explained_pct: float


class FactorAttribution:
    """Decomposes portfolio P&L and risk into PCA factors."""

    FACTOR_NAMES = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]

    @staticmethod
    def compute_factor_exposures(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        spot_curve: SpotCurve,
        spreads_bps: list[float],
        pca_model: PCAModel,
    ) -> np.ndarray:
        """
        Compute factor exposures for each bond.

        Algorithm:
        1. For each bond, compute KR01 at KEY_RATE_TENORS [0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        2. Interpolate KR01 onto PCA_tenors (typically 11 tenors from yield history)
        3. Multiply by PCA factor loadings: exposure[b, f] = kr01_interpolated[b] @ loadings[:, f]

        Args:
            cashflows_list: List of bond cash flows
            notionals: Notional amounts
            spot_curve: Current spot curve
            spreads_bps: Credit spreads (bps)
            pca_model: Fitted PCA model with factor loadings

        Returns:
            shape (n_bonds, n_factors) — factor exposure for each bond to each factor
        """
        if len(cashflows_list) == 0:
            return np.zeros((0, pca_model.factor_loadings.shape[1]))

        n_bonds = len(cashflows_list)
        n_factors = pca_model.factor_loadings.shape[1]

        factor_exposures = np.zeros((n_bonds, n_factors))

        for i, (cf, notional, spread) in enumerate(zip(cashflows_list, notionals, spreads_bps)):
            # Compute KR01 at standard tenors
            kr01_result = KR01Calculator.compute_kr01(cf, spot_curve, notional, spread)

            # Extract KR01 values and tenors
            kr01_tenors = sorted(kr01_result.kr01_by_tenor.keys())
            kr01_values = np.array([kr01_result.kr01_by_tenor[t] for t in kr01_tenors])

            # Interpolate KR01 onto PCA tenors
            # np.interp: interpolate kr01_values at kr01_tenors onto pca_model.tenors
            # left=0, right=0 means extrapolate with zeros
            kr01_interp = np.interp(
                pca_model.tenors,
                kr01_tenors,
                kr01_values,
                left=0.0,
                right=0.0,
            )

            # Compute factor exposure: kr01_interp @ loadings for each factor
            # loadings shape: (n_tenors, n_factors)
            # kr01_interp shape: (n_tenors,)
            # result shape: (n_factors,)
            factor_exposures[i, :] = kr01_interp @ pca_model.factor_loadings

        logger.debug(f"Computed factor exposures for {n_bonds} bonds")

        return factor_exposures

    @staticmethod
    def compute(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        spot_curve: SpotCurve,
        spreads_bps: list[float],
        pca_model: PCAModel,
        curve_change_pct: np.ndarray,
        actual_pnl_by_bond: np.ndarray | None = None,
    ) -> FactorAttributionResult:
        """
        Decompose portfolio P&L into PCA factor contributions.

        Algorithm:
        1. Compute factor exposures (KR01 × PCA loadings)
        2. Project curve change onto factors: realized_scores = PCAModel.project(curve_change)
        3. Compute factor P&L: factor_pnl[b, f] = exposure[b, f] × realized_score[f]
        4. Sum across bonds to get portfolio factor P&L

        Args:
            cashflows_list: List of bond cash flows
            notionals: Notional amounts
            spot_curve: Current spot curve
            spreads_bps: Credit spreads (bps)
            pca_model: Fitted PCA model
            curve_change_pct: Daily yield curve change in percent units (e.g., 0.0425 for 4.25%)
            actual_pnl_by_bond: Actual observed P&L by bond (optional, for residual calculation)

        Returns:
            FactorAttributionResult with full decomposition
        """
        if len(cashflows_list) == 0:
            return FactorAttributionResult(
                portfolio_factor_pnl={},
                per_bond={},
                total_factor_explained=0.0,
                actual_portfolio_pnl=0.0,
                residual_pnl=0.0,
                factor_exposures=np.zeros((0, 3)),
                realized_factor_scores=np.zeros(3),
                explained_pct=0.0,
            )

        # 1. Compute factor exposures
        factor_exposures = FactorAttribution.compute_factor_exposures(
            cashflows_list, notionals, spot_curve, spreads_bps, pca_model
        )

        # 2. Project curve change onto factors
        realized_factor_scores = pca_model.project(curve_change_pct)

        # 3. Compute factor P&L: exposure @ realized_scores
        # factor_exposures shape: (n_bonds, n_factors)
        # realized_factor_scores shape: (n_factors,)
        # result shape: (n_bonds, n_factors)
        bond_factor_pnl = factor_exposures * realized_factor_scores[np.newaxis, :]

        # 4. Portfolio factor P&L (sum across bonds)
        portfolio_factor_pnl_array = bond_factor_pnl.sum(axis=0)

        # 5. Per-bond attribution
        per_bond = {}
        for i, cf in enumerate(cashflows_list):
            factor_pnl_dict = {
                FactorAttribution.FACTOR_NAMES[f]: float(bond_factor_pnl[i, f])
                for f in range(len(FactorAttribution.FACTOR_NAMES))
            }
            total_factor_pnl = bond_factor_pnl[i, :].sum()

            # Actual P&L
            if actual_pnl_by_bond is not None:
                actual_pnl = float(actual_pnl_by_bond[i])
                residual_pnl = actual_pnl - total_factor_pnl
                explained = total_factor_pnl / actual_pnl if actual_pnl != 0 else 0.0
            else:
                actual_pnl = total_factor_pnl
                residual_pnl = 0.0
                explained = 1.0

            per_bond[cf.bond_id] = BondFactorAttribution(
                bond_id=cf.bond_id,
                factor_pnl=factor_pnl_dict,
                total_factor_pnl=total_factor_pnl,
                actual_pnl=actual_pnl,
                residual_pnl=residual_pnl,
                explained_pct=explained,
            )

        # 6. Portfolio-level attribution
        total_factor_explained = portfolio_factor_pnl_array.sum()

        if actual_pnl_by_bond is not None:
            actual_portfolio_pnl = float(actual_pnl_by_bond.sum())
        else:
            actual_portfolio_pnl = total_factor_explained

        residual_pnl = actual_portfolio_pnl - total_factor_explained
        explained_pct = total_factor_explained / actual_portfolio_pnl if actual_portfolio_pnl != 0 else 0.0

        portfolio_factor_pnl = {
            FactorAttribution.FACTOR_NAMES[f]: float(portfolio_factor_pnl_array[f])
            for f in range(len(FactorAttribution.FACTOR_NAMES))
        }

        logger.info(
            f"Factor attribution: PC1={portfolio_factor_pnl[FactorAttribution.FACTOR_NAMES[0]]:.2f}, "
            f"PC2={portfolio_factor_pnl[FactorAttribution.FACTOR_NAMES[1]]:.2f}, "
            f"PC3={portfolio_factor_pnl[FactorAttribution.FACTOR_NAMES[2]]:.2f}, "
            f"Total={total_factor_explained:.2f} (explained {explained_pct*100:.1f}%)"
        )

        return FactorAttributionResult(
            portfolio_factor_pnl=portfolio_factor_pnl,
            per_bond=per_bond,
            total_factor_explained=total_factor_explained,
            actual_portfolio_pnl=actual_portfolio_pnl,
            residual_pnl=residual_pnl,
            factor_exposures=factor_exposures,
            realized_factor_scores=realized_factor_scores,
            explained_pct=explained_pct,
        )
