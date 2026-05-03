"""
Marginal VaR decomposition: break down portfolio VaR into per-bond contributions.

Uses Euler covariance decomposition: Component_VaR_i = Cov(PnL_i, PnL_portfolio) / Std(PnL_portfolio) × z_α
Key property: Σ Component_VaR_i = Total_VaR exactly (within numerical precision)
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pnl.factor_attribution import FactorAttribution
from curves.pca_model import PCAModel
from pricing.cashflow_generator import CashFlowSchedule
from curves.bootstrapper import SpotCurve
from risk.monte_carlo import MonteCarloResult, MonteCarloEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VaRDecomposition:
    """Per-bond VaR decomposition using Euler method."""

    component_var: dict[str, float]  # Covariance-based component VaR
    marginal_var: dict[str, float]  # Component VaR per unit notional
    var_contribution_pct: dict[str, float]  # % of total VaR
    standalone_var: dict[str, float]  # Single-bond VaR (for diversification benefit)
    diversification_benefit: float  # Σ standalone_var - total_var
    total_var: float
    confidence: float


class MarginalVaRCalculator:
    """Decomposes portfolio VaR into per-bond marginal contributions."""

    @staticmethod
    def decompose(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        spot_curve: SpotCurve,
        spreads_bps: list[float],
        pca_model: PCAModel,
        mc_result: MonteCarloResult,
        confidence: float = 0.95,
    ) -> VaRDecomposition:
        """
        Decompose portfolio VaR using Euler covariance method.

        Algorithm:
        1. Use factor exposures to compute per-bond P&L for each scenario:
           per_bond_pnl[b, s] = Σ_f factor_exposure[b, f] × factor_score[s, f]
        2. Compute covariance between each bond's P&L and portfolio P&L
        3. Component_VaR_i = Cov(PnL_i, PnL_portfolio) / Std(PnL_portfolio) × z_α
        4. Marginal_VaR_i = Component_VaR_i / notional_i
        5. For diversification benefit, compute standalone VaR for each bond

        Args:
            cashflows_list: List of bond cash flows
            notionals: Notional amounts
            spot_curve: Current spot curve
            spreads_bps: Credit spreads (bps)
            pca_model: Fitted PCA model
            mc_result: Monte Carlo result with factor_scores
            confidence: Confidence level (default 0.95)

        Returns:
            VaRDecomposition with component and marginal VaR
        """
        if len(cashflows_list) == 0:
            return VaRDecomposition(
                component_var={},
                marginal_var={},
                var_contribution_pct={},
                standalone_var={},
                diversification_benefit=0.0,
                total_var=0.0,
                confidence=confidence,
            )

        n_bonds = len(cashflows_list)
        n_scenarios = mc_result.factor_scores.shape[0]

        # 1. Compute factor exposures (static, same for all scenarios)
        factor_exposures = FactorAttribution.compute_factor_exposures(
            cashflows_list, notionals, spot_curve, spreads_bps, pca_model
        )

        # 2. Per-bond P&L for each scenario: (n_bonds, n_scenarios)
        # factor_scores shape: (n_scenarios, n_factors)
        # factor_exposures shape: (n_bonds, n_factors)
        # per_bond_pnl[b, s] = Σ_f exposures[b, f] × scores[s, f]
        per_bond_pnl = factor_exposures @ mc_result.factor_scores.T  # (n_bonds, n_scenarios)

        # 3. Portfolio P&L (sum across bonds)
        portfolio_pnl = per_bond_pnl.sum(axis=0)  # (n_scenarios,)

        # 4. Portfolio VaR (confidence level)
        var_index = int(np.ceil((1 - confidence) * n_scenarios)) - 1
        var_index = max(0, min(var_index, n_scenarios - 1))
        sorted_portfolio_pnl = np.sort(portfolio_pnl)
        total_var = abs(sorted_portfolio_pnl[var_index])

        # 5. Compute component VaR using Euler covariance decomposition
        portfolio_std = np.std(portfolio_pnl)

        if portfolio_std == 0:
            logger.warning("Portfolio P&L std is zero, cannot compute marginal VaR")
            component_var_dict = {cf.bond_id: 0.0 for cf in cashflows_list}
        else:
            # Euler decomposition: Component_VaR_i = Cov(PnL_i, PnL_portfolio) / Std(portfolio)
            # This is the beta-scaled contribution; it sums to portfolio std (parametric VaR / z_alpha).
            # Scale by (total_empirical_var / parametric_var) so components sum exactly to total_var.
            z_alpha = norm.ppf(confidence)
            parametric_var = z_alpha * portfolio_std

            raw_components = {}
            for i, cf in enumerate(cashflows_list):
                covariance = np.cov(per_bond_pnl[i, :], portfolio_pnl)[0, 1]
                raw_components[cf.bond_id] = (covariance / portfolio_std) * z_alpha

            # Scale so Σ component_var = total_var (Euler property)
            scale = total_var / parametric_var if parametric_var > 0 else 1.0
            component_var_dict = {k: v * scale for k, v in raw_components.items()}

        # 6. Marginal VaR (component VaR per unit notional)
        marginal_var_dict = {}
        for i, cf in enumerate(cashflows_list):
            if notionals[i] != 0:
                marginal_var_dict[cf.bond_id] = abs(component_var_dict[cf.bond_id]) / notionals[i]
            else:
                marginal_var_dict[cf.bond_id] = 0.0

        # 7. VaR contribution % — signed so they sum to 1.0 (Euler property)
        var_contribution_pct_dict = {}
        if total_var != 0:
            for cf in cashflows_list:
                var_contribution_pct_dict[cf.bond_id] = component_var_dict[cf.bond_id] / total_var
        else:
            var_contribution_pct_dict = {cf.bond_id: 0.0 for cf in cashflows_list}

        # 8. Standalone VaR for each bond
        standalone_var_dict = {}
        for i, (cf, notional, spread) in enumerate(zip(cashflows_list, notionals, spreads_bps)):
            standalone_var = MarginalVaRCalculator.standalone_var(
                cf, notional, spot_curve, spread, pca_model, confidence
            )
            standalone_var_dict[cf.bond_id] = standalone_var

        # 9. Diversification benefit
        sum_standalone = sum(standalone_var_dict.values())
        diversification_benefit = sum_standalone - total_var

        logger.info(
            f"VaR Decomposition: Total={total_var:.2f}, "
            f"Standalone_sum={sum_standalone:.2f}, "
            f"Diversification={diversification_benefit:.2f}"
        )

        return VaRDecomposition(
            component_var=component_var_dict,
            marginal_var=marginal_var_dict,
            var_contribution_pct=var_contribution_pct_dict,
            standalone_var=standalone_var_dict,
            diversification_benefit=diversification_benefit,
            total_var=total_var,
            confidence=confidence,
        )

    @staticmethod
    def standalone_var(
        cashflow: CashFlowSchedule,
        notional: float,
        spot_curve: SpotCurve,
        spread_bps: float,
        pca_model: PCAModel,
        confidence: float = 0.95,
        n_scenarios: int = 5000,
    ) -> float:
        """
        Compute single-bond VaR using Monte Carlo.

        Args:
            cashflow: Bond cash flow schedule
            notional: Notional amount
            spot_curve: Current spot curve
            spread_bps: Credit spread (bps)
            pca_model: Fitted PCA model
            confidence: Confidence level
            n_scenarios: Number of Monte Carlo scenarios

        Returns:
            Single-bond VaR in currency units
        """
        try:
            # Run MC for single bond
            mc_engine = MonteCarloEngine(pca_model, spot_curve, n_scenarios=n_scenarios)
            mc_result = mc_engine.simulate([cashflow], [spread_bps], [100.0])

            # Extract P&L and compute VaR
            pnl_dist = mc_result.pnl_distribution
            var_index = int(np.ceil((1 - confidence) * n_scenarios)) - 1
            var_index = max(0, min(var_index, n_scenarios - 1))
            sorted_pnl = np.sort(pnl_dist)
            standalone_var = abs(sorted_pnl[var_index])

            return standalone_var
        except Exception as e:
            logger.warning(f"Failed to compute standalone VaR for {cashflow.bond_id}: {e}")
            return 0.0
