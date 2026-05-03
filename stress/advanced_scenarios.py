"""
Advanced stress scenarios: butterfly, twist, rolldown, reverse stress, historical.

These are production-grade scenarios that professional portfolio managers actually use.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config.settings import settings
from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import price_batch
from pricing.cashflow_generator import CashFlow, CashFlowSchedule
from stress.scenario_runner import StressResult, ScenarioRunner

logger = logging.getLogger(__name__)


@dataclass
class AdvancedStressResult(StressResult):
    """Extended stress result with additional metadata."""

    scenario_type: str = "hypothetical"  # "hypothetical" | "historical" | "reverse"
    per_tenor_shocks: dict[str, float] = field(default_factory=dict)  # Shock by tenor
    per_bond_impact: dict[str, float] = field(default_factory=dict)  # Impact by bond
    reverse_stress_trigger: float | None = None  # Parallel shift at which limit breached


class AdvancedScenarioRunner:
    """Runs advanced stress scenarios."""

    # Mapping from DGS series to tenor in years
    DGS_TO_TENOR = {
        "DGS1MO": 1.0 / 12,
        "DGS3MO": 0.25,
        "DGS6MO": 0.5,
        "DGS1": 1.0,
        "DGS2": 2.0,
        "DGS3": 3.0,
        "DGS5": 5.0,
        "DGS7": 7.0,
        "DGS10": 10.0,
        "DGS20": 20.0,
        "DGS30": 30.0,
    }

    @staticmethod
    def run_butterfly_scenario(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
    ) -> AdvancedStressResult:
        """
        Butterfly scenario: 2Y +100bps, 5Y -50bps, 10Y +50bps.

        Belly outperformance (curvature trade).
        """
        yield_shocks = {
            "2Y": 100.0,
            "5Y": -50.0,
            "10Y": 50.0,
        }

        result = ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name="Butterfly (2Y+100, 5Y-50, 10Y+50)",
            scenario_id="butterfly",
            yield_curve_shocks=yield_shocks,
        )

        return AdvancedStressResult(
            **result.__dict__,
            scenario_type="hypothetical",
            per_tenor_shocks=yield_shocks,
        )

    @staticmethod
    def run_twist_scenario(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
    ) -> AdvancedStressResult:
        """
        Twist scenario: 2Y +150bps, 10Y +50bps, 30Y -50bps.

        Aggressive curve flattening (long end underperforms).
        """
        yield_shocks = {
            "2Y": 150.0,
            "10Y": 50.0,
            "30Y": -50.0,
        }

        result = ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name="Twist (2Y+150, 10Y+50, 30Y-50)",
            scenario_id="twist",
            yield_curve_shocks=yield_shocks,
        )

        return AdvancedStressResult(
            **result.__dict__,
            scenario_type="hypothetical",
            per_tenor_shocks=yield_shocks,
        )

    @staticmethod
    def run_rolldown_scenario(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
        horizon_days: int = 30,
    ) -> AdvancedStressResult:
        """
        Rolldown scenario: bonds move down the curve, accrue carry.

        Algorithm:
        1. Shift each bond's tenor -= horizon_days/365
        2. Reprice on same curve (benefits from curve shape)
        3. Add carry P&L (coupon accrual)

        Args:
            cashflows_list: Bond cash flows
            notionals: Notional amounts
            current_curve: Current spot curve
            current_spreads_bps: Current spreads
            horizon_days: Time horizon (default 30 days)

        Returns:
            AdvancedStressResult with rolldown P&L
        """
        # Current prices
        baseline_prices = price_batch(cashflows_list, current_curve, current_spreads_bps)
        baseline_value = np.sum(baseline_prices * np.array(notionals) / 100.0)

        # Shift cash flows forward in time
        # For each cash flow, subtract horizon_days from its time to maturity
        tenor_shift = horizon_days / 365.0

        # Create shifted cash flows — reduce each CashFlow tenor by horizon_days/365
        shifted_cashflows = []
        for cf in cashflows_list:
            shifted_flows = [
                CashFlow(date=cfl.date, amount=cfl.amount,
                         tenor=max(0.0, cfl.tenor - tenor_shift), cf_type=cfl.cf_type)
                for cfl in cf.cash_flows
            ]
            shifted_cf = CashFlowSchedule(
                bond_id=cf.bond_id,
                face_value=cf.face_value,
                coupon_rate=cf.coupon_rate,
                coupon_frequency=cf.coupon_frequency,
                cash_flows=shifted_flows,
            )
            shifted_cashflows.append(shifted_cf)

        # Reprice on same curve (rolldown benefit from curve shape)
        rolled_prices = price_batch(shifted_cashflows, current_curve, current_spreads_bps)

        # Add carry P&L (daily coupon × notional × days)
        carry_pnl = 0.0
        for i, cf in enumerate(cashflows_list):
            daily_carry = (cf.face_value * cf.coupon_rate * notionals[i]) / 365.0
            carry_pnl += daily_carry * horizon_days

        # Total P&L = price rolldown + carry
        rolled_value = np.sum(rolled_prices * np.array(notionals) / 100.0)
        impact_absolute = (rolled_value - baseline_value) + carry_pnl
        impact_pct = (impact_absolute / baseline_value) * 100.0 if baseline_value != 0 else 0.0

        # Worst hit bond
        price_impacts = (rolled_prices - baseline_prices) / baseline_prices * 100.0
        worst_idx = np.argmin(price_impacts)
        worst_bond_id = cashflows_list[worst_idx].bond_id
        worst_bond_impact_pct = price_impacts[worst_idx]

        return AdvancedStressResult(
            scenario_name=f"Rolldown ({horizon_days}d)",
            scenario_id="rolldown",
            portfolio_value_baseline=baseline_value,
            portfolio_value_stress=rolled_value + carry_pnl,
            impact_absolute=impact_absolute,
            impact_pct=impact_pct,
            worst_bond_id=worst_bond_id,
            worst_bond_impact_pct=worst_bond_impact_pct,
            scenario_type="hypothetical",
        )

    @staticmethod
    def run_reverse_stress(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
        portfolio_value: float,
        var_limit_pct: float = 0.02,
    ) -> AdvancedStressResult:
        """
        Reverse stress: find parallel shift where portfolio loss = VaR limit.

        Uses scipy.optimize.brentq to find the shift magnitude.

        Args:
            cashflows_list: Bond cash flows
            notionals: Notional amounts
            current_curve: Current spot curve
            current_spreads_bps: Current spreads
            portfolio_value: Current portfolio value
            var_limit_pct: VaR limit as % of portfolio (default 2%)

        Returns:
            AdvancedStressResult with reverse stress trigger point
        """
        var_limit_abs = portfolio_value * (var_limit_pct / 100.0)

        def portfolio_impact(parallel_shift_bps: float) -> float:
            """Compute portfolio impact for given parallel shift."""
            try:
                shifted_rates = current_curve.spot_rates + (parallel_shift_bps / 10000.0)
                shifted_curve = SpotCurve(
                    tenors=current_curve.tenors,
                    spot_rates=shifted_rates,
                    as_of_date=current_curve.as_of_date,
                )

                baseline_prices = price_batch(cashflows_list, current_curve, current_spreads_bps)
                baseline_value = np.sum(baseline_prices * np.array(notionals) / 100.0)

                shifted_prices = price_batch(cashflows_list, shifted_curve, current_spreads_bps)
                shifted_value = np.sum(shifted_prices * np.array(notionals) / 100.0)

                impact = shifted_value - baseline_value
                return impact - (-var_limit_abs)  # We want impact = -var_limit_abs
            except Exception as e:
                logger.warning(f"Error in reverse stress calculation: {e}")
                return 0.0

        # Find the trigger point using Brentq
        try:
            # Search in range [-500, 500] bps
            trigger_bps = brentq(portfolio_impact, -500, 500, xtol=1e-2)
        except ValueError:
            # If no root found in range, use edge of range
            if portfolio_impact(-500) < 0:
                trigger_bps = -500
            else:
                trigger_bps = 500

        # Compute actual stress at trigger point
        shifted_rates = current_curve.spot_rates + (trigger_bps / 10000.0)
        shifted_curve = SpotCurve(
            tenors=current_curve.tenors,
            spot_rates=shifted_rates,
            as_of_date=current_curve.as_of_date,
        )

        baseline_prices = price_batch(cashflows_list, current_curve, current_spreads_bps)
        baseline_value = np.sum(baseline_prices * np.array(notionals) / 100.0)

        shifted_prices = price_batch(cashflows_list, shifted_curve, current_spreads_bps)
        shifted_value = np.sum(shifted_prices * np.array(notionals) / 100.0)

        impact_absolute = shifted_value - baseline_value
        impact_pct = (impact_absolute / baseline_value) * 100.0 if baseline_value != 0 else 0.0

        # Worst hit bond
        price_impacts = (shifted_prices - baseline_prices) / baseline_prices * 100.0
        worst_idx = np.argmin(price_impacts)
        worst_bond_id = cashflows_list[worst_idx].bond_id
        worst_bond_impact_pct = price_impacts[worst_idx]

        logger.info(f"Reverse stress: {trigger_bps:.1f}bps parallel shift breaches {var_limit_pct:.1f}% limit")

        return AdvancedStressResult(
            scenario_name=f"Reverse Stress ({var_limit_pct:.1f}% limit)",
            scenario_id="reverse_stress",
            portfolio_value_baseline=baseline_value,
            portfolio_value_stress=shifted_value,
            impact_absolute=impact_absolute,
            impact_pct=impact_pct,
            worst_bond_id=worst_bond_id,
            worst_bond_impact_pct=worst_bond_impact_pct,
            scenario_type="reverse",
            reverse_stress_trigger=trigger_bps,
        )

    @staticmethod
    def run_historical_scenario(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
        scenario_name: str,
        data_file: str | Path,
        aggregation: str = "total",
    ) -> AdvancedStressResult:
        """
        Historical scenario: load crisis CSV, apply yield changes to curve.

        Args:
            cashflows_list: Bond cash flows
            notionals: Notional amounts
            current_curve: Current spot curve
            current_spreads_bps: Current spreads
            scenario_name: Name of scenario (e.g., "GFC 2008")
            data_file: Path to crisis CSV (columns: DGS1MO, DGS3MO, ..., DGS30)
            aggregation: How to aggregate multiple days:
                - "total": sum of all daily changes
                - "max_1day": maximum single day loss
                - "worst_5day": sum of 5 worst days

        Returns:
            AdvancedStressResult with historical impact
        """
        data_file = Path(data_file)
        if not data_file.exists():
            logger.warning(f"Historical data file not found: {data_file}")
            # Return zero impact
            baseline_prices = price_batch(cashflows_list, current_curve, current_spreads_bps)
            baseline_value = np.sum(baseline_prices * np.array(notionals) / 100.0)
            return AdvancedStressResult(
                scenario_name=scenario_name,
                scenario_id="historical",
                portfolio_value_baseline=baseline_value,
                portfolio_value_stress=baseline_value,
                impact_absolute=0.0,
                impact_pct=0.0,
                worst_bond_id="N/A",
                worst_bond_impact_pct=0.0,
                scenario_type="historical",
            )

        # Load crisis data
        df = pd.read_csv(data_file)

        # Compute aggregated shock
        if aggregation == "total":
            # Sum all daily changes
            shock_dict = {}
            for col in df.columns:
                if col.startswith("DGS"):
                    shock_dict[col] = df[col].sum()
        elif aggregation == "max_1day":
            # Maximum single day loss
            daily_impacts = []
            for idx in range(len(df)):
                daily_df = df.iloc[idx:idx+1]
                daily_impacts.append(daily_df.iloc[:, 1:].sum(axis=1).values[0])
            max_idx = np.argmin(daily_impacts)
            shock_dict = {}
            for col in df.columns:
                if col.startswith("DGS"):
                    shock_dict[col] = df.iloc[max_idx][col]
        elif aggregation == "worst_5day":
            # Sum of 5 worst days
            daily_impacts = []
            for idx in range(len(df)):
                daily_df = df.iloc[idx:idx+1]
                daily_impacts.append(daily_df.iloc[:, 1:].sum(axis=1).values[0])
            worst_indices = np.argsort(daily_impacts)[:5]
            shock_dict = {}
            for col in df.columns:
                if col.startswith("DGS"):
                    shock_dict[col] = df.iloc[worst_indices][col].sum()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Convert DGS shocks to tenor shocks
        # DGS values are in pct-point units (0.007 = 0.7bps), multiply by 100 for bps
        yield_curve_shocks = {}
        for dgs_code, shock_decimal in shock_dict.items():
            if dgs_code in AdvancedScenarioRunner.DGS_TO_TENOR:
                tenor = AdvancedScenarioRunner.DGS_TO_TENOR[dgs_code]
                shock_bps = shock_decimal * 100.0  # pct-point → basis points
                tenor_str = f"{tenor:.1f}Y" if tenor >= 1 else f"{tenor*12:.0f}M"
                yield_curve_shocks[tenor_str] = shock_bps

        # Run scenario
        result = ScenarioRunner.run_hypothetical_scenario(
            cashflows_list,
            notionals,
            current_curve,
            current_spreads_bps,
            scenario_name=scenario_name,
            scenario_id="historical",
            yield_curve_shocks=yield_curve_shocks,
        )

        return AdvancedStressResult(
            **result.__dict__,
            scenario_type="historical",
            per_tenor_shocks=yield_curve_shocks,
        )

    @staticmethod
    def run_all_advanced(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        current_curve: SpotCurve,
        current_spreads_bps: list[float],
        portfolio_value: float,
    ) -> list[AdvancedStressResult]:
        """
        Run all advanced scenarios: butterfly, twist, rolldown, reverse stress, historical.

        Returns:
            List of AdvancedStressResult, sorted by severity (worst first)
        """
        scenarios = []

        # Hypothetical scenarios
        scenarios.append(
            AdvancedScenarioRunner.run_butterfly_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps
            )
        )
        scenarios.append(
            AdvancedScenarioRunner.run_twist_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps
            )
        )
        scenarios.append(
            AdvancedScenarioRunner.run_rolldown_scenario(
                cashflows_list, notionals, current_curve, current_spreads_bps, horizon_days=30
            )
        )
        scenarios.append(
            AdvancedScenarioRunner.run_reverse_stress(
                cashflows_list, notionals, current_curve, current_spreads_bps, portfolio_value
            )
        )

        # Historical scenarios
        historical_dir = settings.SAMPLE_DATA_PATH
        for crisis_file in ["crisis_gfc_2008.csv", "crisis_covid_2020.csv", "crisis_rates_2022.csv"]:
            crisis_path = historical_dir / crisis_file
            if crisis_path.exists():
                scenario_name = {
                    "crisis_gfc_2008.csv": "GFC 2008 (Total)",
                    "crisis_covid_2020.csv": "COVID-19 2020 (Total)",
                    "crisis_rates_2022.csv": "Rate Shock 2022 (Total)",
                }.get(crisis_file, f"Historical ({crisis_file})")

                scenarios.append(
                    AdvancedScenarioRunner.run_historical_scenario(
                        cashflows_list,
                        notionals,
                        current_curve,
                        current_spreads_bps,
                        scenario_name=scenario_name,
                        data_file=crisis_path,
                        aggregation="total",
                    )
                )

        # Sort by severity (worst first)
        scenarios.sort(key=lambda s: s.impact_pct)

        logger.info(f"Ran {len(scenarios)} advanced scenarios")

        return scenarios
