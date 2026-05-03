"""
Brinson-Fachler attribution: compare portfolio vs benchmark, decompose excess return.

Decomposes active return into:
- Allocation effect: did we overweight/underweight high-return buckets?
- Selection effect: did our credit bonds outperform equivalent Treasuries?
- Interaction effect: combined allocation + selection
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config.settings import settings
from curves.bootstrapper import bootstrap_spot_curve, SpotCurve
from pricing.bond_pricer import price_batch
from pricing.cashflow_generator import generate_cashflow_schedule, DayCountMethod, CashFlowSchedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BondBrinsonResult:
    """Attribution effect for a single bond."""

    bond_id: str
    benchmark_bucket: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    benchmark_total_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    active_effect: float


@dataclass
class BrinsonFachlerResult:
    """Full Brinson-Fachler attribution result."""

    attribution_date: date
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    total_active_return: float
    portfolio_total_return: float
    benchmark_total_return: float
    per_bond: dict[str, BondBrinsonResult] = field(default_factory=dict)
    explained_pct: float = 0.0


class BrinsonFachler:
    """Attribution analysis: portfolio vs benchmark returns."""

    @classmethod
    def load_benchmark(
        cls,
        spot_curve: SpotCurve,
        as_of_date: date,
        config_path: str | None = None,
    ) -> tuple[list[CashFlowSchedule], list[float], list[float]]:
        """
        Load benchmark bonds from config.

        Args:
            spot_curve: Current spot curve
            as_of_date: As-of date for cash flow generation
            config_path: Path to benchmark config (default config/benchmark.yaml)

        Returns:
            (cashflows_list, weights, spreads_bps)
        """
        if config_path is None:
            config_path = settings.BENCHMARK_CONFIG_PATH

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {config_path}")

        with open(config_path, 'r') as f:
            benchmark_config = yaml.safe_load(f)

        benchmark = benchmark_config['benchmark']
        cashflows_list = []
        weights = []
        spreads_bps = []

        total_notional = benchmark.get('total_notional', 500_000_000)

        for bond_config in benchmark['bonds']:
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
            weights.append(bond_config['weight'])
            spreads_bps.append(bond_config.get('credit_spread_bps', 0.0))

        return cashflows_list, weights, spreads_bps

    @staticmethod
    def compute(
        portfolio_cashflows: list[CashFlowSchedule],
        portfolio_weights: list[float],
        portfolio_spreads: list[float],
        portfolio_bond_ids: list[str],
        benchmark_cashflows: list[CashFlowSchedule],
        benchmark_weights: list[float],
        benchmark_spreads: list[float],
        benchmark_bond_ids: list[str],
        spot_curve_today: SpotCurve,
        spot_curve_yesterday: SpotCurve,
        attribution_date: date = None,
    ) -> BrinsonFachlerResult:
        """
        Compute Brinson-Fachler attribution.

        Algorithm:
        1. Compute returns for each portfolio and benchmark bond
        2. Map portfolio bonds to benchmark buckets (by tenor)
        3. For each bucket:
           - allocation_effect = (w_p - w_b) × (R_b - R_b_total)
           - selection_effect = w_b × (R_p - R_b)
           - interaction_effect = (w_p - w_b) × (R_p - R_b)

        Args:
            portfolio_cashflows: Portfolio bond cash flows
            portfolio_weights: Portfolio weights (sum to 1.0)
            portfolio_spreads: Portfolio spreads (bps)
            portfolio_bond_ids: Portfolio bond IDs
            benchmark_cashflows: Benchmark bond cash flows
            benchmark_weights: Benchmark weights (sum to 1.0)
            benchmark_spreads: Benchmark spreads (bps)
            benchmark_bond_ids: Benchmark bond IDs
            spot_curve_today: Today's spot curve
            spot_curve_yesterday: Yesterday's spot curve
            attribution_date: Attribution date (default today)

        Returns:
            BrinsonFachlerResult with full attribution
        """
        if attribution_date is None:
            attribution_date = date.today()

        # 1. Compute returns
        portfolio_prices_today = price_batch(portfolio_cashflows, spot_curve_today, portfolio_spreads)
        portfolio_prices_yesterday = price_batch(
            portfolio_cashflows, spot_curve_yesterday, portfolio_spreads
        )
        portfolio_returns = (portfolio_prices_today - portfolio_prices_yesterday) / portfolio_prices_yesterday

        benchmark_prices_today = price_batch(benchmark_cashflows, spot_curve_today, benchmark_spreads)
        benchmark_prices_yesterday = price_batch(
            benchmark_cashflows, spot_curve_yesterday, benchmark_spreads
        )
        benchmark_returns = (benchmark_prices_today - benchmark_prices_yesterday) / benchmark_prices_yesterday

        # Portfolio and benchmark total returns
        portfolio_total_return = np.sum(
            np.array(portfolio_returns) * np.array(portfolio_weights)
        )
        benchmark_total_return = np.sum(
            np.array(benchmark_returns) * np.array(benchmark_weights)
        )

        # 2. Attribution per bond
        per_bond = {}
        total_allocation = 0.0
        total_selection = 0.0
        total_interaction = 0.0

        # Create mapping of portfolio bonds to benchmark buckets
        for i, (p_id, p_weight, p_return) in enumerate(
            zip(portfolio_bond_ids, portfolio_weights, portfolio_returns)
        ):
            # Find matching benchmark bond (by tenor/ID pattern)
            matching_idx = None
            for j, (b_id, b_weight, b_return) in enumerate(
                zip(benchmark_bond_ids, benchmark_weights, benchmark_returns)
            ):
                # Match by tenor from tenor field if available, or by bucket name
                if p_id.replace("_", "") in b_id.replace("_", "") or \
                   b_id.split("_")[0] in p_id:
                    matching_idx = j
                    break

            # If no exact match, find closest by tenor value if available
            if matching_idx is None and len(benchmark_bond_ids) > 0:
                matching_idx = 0  # Default to first benchmark bond

            # Compute effects
            b_weight = benchmark_weights[matching_idx] if matching_idx is not None else 0.0
            b_return = benchmark_returns[matching_idx] if matching_idx is not None else 0.0
            b_bucket = benchmark_bond_ids[matching_idx] if matching_idx is not None else "Unknown"

            # Allocation: did we overweight high-return buckets?
            allocation = (p_weight - b_weight) * (b_return - benchmark_total_return)

            # Selection: did our bond outperform the benchmark equivalent?
            selection = b_weight * (p_return - b_return)

            # Interaction: combined effect
            interaction = (p_weight - b_weight) * (p_return - b_return)

            # Active effect: sum of allocation + selection + interaction
            active = allocation + selection + interaction

            per_bond[p_id] = BondBrinsonResult(
                bond_id=p_id,
                benchmark_bucket=b_bucket,
                portfolio_weight=p_weight,
                benchmark_weight=b_weight,
                portfolio_return=float(p_return),
                benchmark_return=float(b_return),
                benchmark_total_return=float(benchmark_total_return),
                allocation_effect=allocation,
                selection_effect=selection,
                interaction_effect=interaction,
                active_effect=active,
            )

            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction

        total_active_return = portfolio_total_return - benchmark_total_return
        explained = abs(total_allocation + total_selection + total_interaction)
        explained_pct = explained / abs(total_active_return) if total_active_return != 0 else 0.0

        logger.info(
            f"Brinson-Fachler: Allocation={total_allocation:.4f}, "
            f"Selection={total_selection:.4f}, Interaction={total_interaction:.4f}, "
            f"Active={total_active_return:.4f}"
        )

        return BrinsonFachlerResult(
            attribution_date=attribution_date,
            total_allocation_effect=total_allocation,
            total_selection_effect=total_selection,
            total_interaction_effect=total_interaction,
            total_active_return=total_active_return,
            portfolio_total_return=float(portfolio_total_return),
            benchmark_total_return=float(benchmark_total_return),
            per_bond=per_bond,
            explained_pct=explained_pct,
        )
