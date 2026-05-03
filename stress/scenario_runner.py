"""
Stress testing scenario runner.

Runs hypothetical and historical scenarios, computes portfolio impact.
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import price_batch
from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Result of a single stress test scenario."""

    scenario_name: str
    scenario_id: str
    portfolio_value_baseline: float
    portfolio_value_stress: float
    impact_absolute: float
    impact_pct: float
    worst_bond_id: str
    worst_bond_impact_pct: float


class ScenarioRunner:
    """Runs stress testing scenarios on portfolio."""

    @staticmethod
    def run_hypothetical_scenario(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
        scenario_name: str,
        scenario_id: str,
        yield_curve_parallel_bps: float = 0.0,
        yield_curve_shocks: dict[str, float] | None = None,
        credit_spread_bps: float = 0.0,
        credit_spread_by_rating: dict[str, float] | None = None,
    ) -> StressResult:
        """
        Run hypothetical stress scenario (parallel shift, steepener, spread shock, etc).

        Args:
            cashflows_list: List of bond cash flows
            notionals: Notional amounts
            current_curve: Current spot curve
            current_spreads_bps: Current spreads (bps)
            scenario_name: Name of scenario
            scenario_id: ID of scenario
            yield_curve_parallel_bps: Parallel shift to all tenors (bps)
            yield_curve_shocks: Custom shocks by tenor {"2Y": 50, "10Y": -20, ...}
            credit_spread_bps: Credit spread parallel shift (bps)
            credit_spread_by_rating: Rating-specific spread shocks {"AAA": 100, "BB": 200, ...}

        Returns:
            StressResult with portfolio impact
        """
        # Current prices
        baseline_prices = price_batch(cashflows_list, current_curve, current_spreads_bps)
        baseline_value = np.sum(baseline_prices * np.array(notionals) / 100.0)

        # Build shocked curve
        if yield_curve_parallel_bps != 0.0:
            shocked_rates = current_curve.spot_rates + (yield_curve_parallel_bps / 10000.0)
        else:
            shocked_rates = current_curve.spot_rates.copy()

        # Apply custom tenor shocks if provided
        if yield_curve_shocks:
            for tenor, shock_bps in yield_curve_shocks.items():
                if isinstance(tenor, (int, float)):
                    tenor_float = float(tenor)
                elif str(tenor).endswith("M"):
                    tenor_float = float(str(tenor).rstrip("M")) / 12.0
                else:
                    tenor_float = float(str(tenor).rstrip("Y"))
                # Find nearest tenor in curve
                idx = np.argmin(np.abs(current_curve.tenors - tenor_float))
                shocked_rates[idx] += shock_bps / 10000.0

        # Create shocked curve
        from curves.bootstrapper import SpotCurve

        stressed_curve = SpotCurve(
            tenors=current_curve.tenors,
            spot_rates=shocked_rates,
            as_of_date=current_curve.as_of_date,
        )

        # Apply credit spread shocks
        if isinstance(current_spreads_bps, (int, float)):
            shocked_spreads = [current_spreads_bps + credit_spread_bps] * len(cashflows_list)
        else:
            shocked_spreads = [s + credit_spread_bps for s in current_spreads_bps]

        # Apply rating-specific shocks if provided
        if credit_spread_by_rating:
            for i, cf in enumerate(cashflows_list):
                if hasattr(cf, "rating") and cf.rating in credit_spread_by_rating:
                    shocked_spreads[i] += credit_spread_by_rating[cf.rating]

        # Stressed prices
        stressed_prices = price_batch(cashflows_list, stressed_curve, shocked_spreads)
        stressed_value = np.sum(stressed_prices * np.array(notionals) / 100.0)

        # Impact
        impact_absolute = stressed_value - baseline_value
        impact_pct = (impact_absolute / baseline_value) * 100.0 if baseline_value != 0 else 0.0

        # Worst hit bond
        price_impacts = (stressed_prices - baseline_prices) / baseline_prices * 100.0
        worst_idx = np.argmin(price_impacts)
        worst_bond_id = cashflows_list[worst_idx].bond_id
        worst_bond_impact_pct = price_impacts[worst_idx]

        result = StressResult(
            scenario_name=scenario_name,
            scenario_id=scenario_id,
            portfolio_value_baseline=baseline_value,
            portfolio_value_stress=stressed_value,
            impact_absolute=impact_absolute,
            impact_pct=impact_pct,
            worst_bond_id=worst_bond_id,
            worst_bond_impact_pct=worst_bond_impact_pct,
        )

        logger.info(f"Scenario '{scenario_name}': {impact_pct:.2f}% impact")

        return result

    @staticmethod
    def run_all_scenarios(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
    ) -> list[StressResult]:
        """
        Run all standard scenarios: parallel shifts and spread shocks.

        Returns:
            List of StressResult, sorted by impact (worst first)
        """
        scenarios = [
            StressResult.create_parallel_up_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps, bump_bps=200
            ),
            StressResult.create_parallel_down_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps, bump_bps=100
            ),
            StressResult.create_steepener_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps
            ),
            StressResult.create_spread_blowout_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps
            ),
        ]

        # Sort by severity (worst impact first)
        scenarios.sort(key=lambda s: s.impact_pct)

        return scenarios

    @staticmethod
    def create_parallel_up_scenario(
        cashflows_list,
        notionals,
        current_curve,
        current_spreads_bps,
        bump_bps: float = 200,
    ) -> StressResult:
        """Parallel +200bps scenario."""
        return ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name=f"Parallel +{bump_bps}bps",
            scenario_id="parallel_up",
            yield_curve_parallel_bps=bump_bps,
        )

    @staticmethod
    def create_parallel_down_scenario(
        cashflows_list,
        notionals,
        current_curve,
        current_spreads_bps,
        bump_bps: float = 100,
    ) -> StressResult:
        """Parallel -100bps scenario."""
        return ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name=f"Parallel -{bump_bps}bps",
            scenario_id="parallel_down",
            yield_curve_parallel_bps=-bump_bps,
        )

    @staticmethod
    def create_steepener_scenario(
        cashflows_list,
        notionals,
        current_curve,
        current_spreads_bps,
    ) -> StressResult:
        """Steepener: short end up 150bps, long end down 50bps."""
        shocks = {
            "2Y": 150,
            "5Y": 50,
            "10Y": -50,
            "30Y": -50,
        }
        return ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name="Curve Steepener (2Y +150bps, 30Y -50bps)",
            scenario_id="steepener",
            yield_curve_shocks=shocks,
        )

    @staticmethod
    def create_spread_blowout_scenario(
        cashflows_list,
        notionals,
        current_curve,
        current_spreads_bps,
    ) -> StressResult:
        """Credit spread blowout: spreads widen by rating."""
        spreads_by_rating = {
            "AAA": 100,
            "AA": 200,
            "A": 350,
            "BBB": 500,
        }
        return ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name="Credit Spread Blowout",
            scenario_id="spread_blowout",
            credit_spread_by_rating=spreads_by_rating,
        )

    @staticmethod
    def scenario_summary_table(results: list[StressResult]) -> pd.DataFrame:
        """
        Create summary table of scenario results.

        Args:
            results: List of StressResult

        Returns:
            DataFrame with scenario summary
        """
        records = []

        for result in results:
            records.append({
                "Scenario": result.scenario_name,
                "Baseline (Rs Cr)": result.portfolio_value_baseline / 1e7,
                "Stressed (Rs Cr)": result.portfolio_value_stress / 1e7,
                "Impact (Rs Cr)": result.impact_absolute / 1e7,
                "Impact (%)": result.impact_pct,
                "Worst Bond": result.worst_bond_id,
                "Worst Bond Impact (%)": result.worst_bond_impact_pct,
            })

        df = pd.DataFrame(records)
        return df
