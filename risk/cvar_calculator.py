"""CVaR/Expected Shortfall calculation."""

import logging

import numpy as np

from risk.monte_carlo import MonteCarloResult
from risk.var_calculator import VaRCalculator

logger = logging.getLogger(__name__)


class CVaRCalculator:
    """Computes Conditional Value at Risk (Expected Shortfall)."""

    @staticmethod
    def monte_carlo_cvar(
        mc_result: MonteCarloResult,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute CVaR/ES as average of losses beyond VaR threshold.

        CVaR = average(losses where loss > VaR)

        Also called Expected Shortfall (ES).

        Args:
            mc_result: Monte Carlo simulation result
            confidence: Confidence level

        Returns:
            CVaR in currency units
        """
        # Compute VaR first
        var = VaRCalculator.monte_carlo_var(mc_result, confidence)

        # Find all scenarios with losses >= VaR
        losses = -mc_result.pnl_distribution
        tail_losses = losses[losses >= var]

        if len(tail_losses) == 0:
            # Edge case: no losses exceed VaR (very good simulation)
            return var

        cvar = tail_losses.mean()

        logger.debug(f"CVaR {confidence*100:.0f}%: {cvar:.2f} (VaR: {var:.2f})")

        return cvar

    @staticmethod
    def tail_ratio(var: float, cvar: float) -> float:
        """
        Compute tail ratio: CVaR / VaR.

        Indicates fatness of tail.
        - ~1.0: thin tail (normal-like)
        - >1.5: fat tail (dangerous)

        Args:
            var: Value at Risk
            cvar: Conditional Value at Risk

        Returns:
            Ratio (CVaR / VaR)
        """
        if var == 0:
            return 1.0

        return cvar / var

    @staticmethod
    def cvar_multiple_confidence_levels(
        mc_result: MonteCarloResult,
        confidence_levels: list[float] = [0.90, 0.95, 0.99],
    ) -> dict[float, float]:
        """
        Compute CVaR at multiple confidence levels.

        Args:
            mc_result: Monte Carlo result
            confidence_levels: Confidence levels to compute

        Returns:
            Dict mapping confidence level to CVaR
        """
        cvar_dict = {}

        for conf in confidence_levels:
            cvar = CVaRCalculator.monte_carlo_cvar(mc_result, confidence=conf)
            cvar_dict[conf] = cvar

        return cvar_dict

    @staticmethod
    def frtb_expected_shortfall(
        mc_result: MonteCarloResult,
        confidence: float = 0.975,
    ) -> float:
        """
        FRTB-compliant Expected Shortfall at 97.5% confidence.

        Basel post-2019 (BCBS 352) uses ES instead of VaR as capital measure.

        Args:
            mc_result: Monte Carlo result
            confidence: Confidence level (default 97.5% for FRTB)

        Returns:
            FRTB Expected Shortfall
        """
        es = CVaRCalculator.monte_carlo_cvar(mc_result, confidence=confidence)

        logger.info(f"FRTB Expected Shortfall ({confidence*100:.1f}%): {es:.2f}")

        return es
