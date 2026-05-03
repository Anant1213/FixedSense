"""
P&L attribution: decompose daily P&L into carry, rate, spread, residual.

This is what portfolio managers review every morning.
Formula: carry + rate + spread + residual = total_pnl (within Rs 1)
"""

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from curves.bootstrapper import SpotCurve
from greeks.duration import DurationCalculator
from greeks.spread_duration import SpreadDurationCalculator
from pricing.bond_pricer import BondPricer, price_batch
from pricing.cashflow_generator import CashFlowSchedule
from pricing.ytm_solver import solve_ytm

logger = logging.getLogger(__name__)


@dataclass
class BondAttributionResult:
    """P&L attribution for a single bond."""

    bond_id: str
    notional: float
    carry_pnl: float
    rate_pnl: float
    spread_pnl: float
    residual_pnl: float
    total_pnl: float
    explained_pct: float  # (carry + rate + spread) / total

    def __str__(self) -> str:
        """Pretty print attribution."""
        return (
            f"{self.bond_id:12s} | Carry: {self.carry_pnl:10.2f} | Rate: {self.rate_pnl:10.2f} | "
            f"Spread: {self.spread_pnl:10.2f} | Residual: {self.residual_pnl:10.2f} | "
            f"Total: {self.total_pnl:10.2f} ({self.explained_pct:.1%} explained)"
        )


@dataclass
class AttributionResult:
    """Full portfolio P&L attribution result."""

    date: date
    total_pnl: float
    carry_pnl: float
    rate_pnl: float
    spread_pnl: float
    residual_pnl: float
    per_bond: dict[str, BondAttributionResult] = field(default_factory=dict)
    explain_pct: float = 0.0  # How much of total is explained by components

    def summary(self) -> str:
        """Summary table."""
        lines = [
            f"Portfolio P&L Attribution for {self.date}",
            "=" * 80,
            f"Total P&L:      {self.total_pnl:15.2f}",
            f"  Carry:        {self.carry_pnl:15.2f} ({self.carry_pnl/self.total_pnl*100:6.1f}%)",
            f"  Rate:         {self.rate_pnl:15.2f} ({self.rate_pnl/self.total_pnl*100:6.1f}%)",
            f"  Spread:       {self.spread_pnl:15.2f} ({self.spread_pnl/self.total_pnl*100:6.1f}%)",
            f"  Residual:     {self.residual_pnl:15.2f} ({self.residual_pnl/self.total_pnl*100:6.1f}%)",
            "-" * 80,
            f"Explained:      {self.explain_pct*100:14.1f}%",
        ]
        return "\n".join(lines)


class PnLAttribution:
    """Decomposes daily portfolio P&L into carry, rate, spread, residual."""

    @staticmethod
    def compute_daily_attribution(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
        yesterday_prices: np.ndarray,
        yesterday_curve: SpotCurve,
        yesterday_spreads: list[float],
        today_curve: SpotCurve,
        today_spreads: list[float],
        attribution_date: date = None,
    ) -> AttributionResult:
        """
        Decompose daily P&L into carry, rate, spread, residual.

        Algorithm:
        1. CARRY: coupon earned each day = (annual_coupon * notional) / 365
        2. RATE: effect of yield curve change on bond values
           rate_pnl = -dv01 * (yield_change_bps)
        3. SPREAD: effect of OAS spread widening/tightening
           spread_pnl = -spread_dv01 * (spread_change_bps)
        4. RESIDUAL: total - carry - rate - spread
           Captures: convexity effect, theta, model error, liquidity

        Args:
            cashflows_list: List of bond cash flows
            notionals: Notional amounts (Rs)
            yesterday_prices: Yesterday's bond prices
            yesterday_curve: Yesterday's spot curve
            yesterday_spreads: Yesterday's spreads (bps)
            today_curve: Today's spot curve
            today_spreads: Today's spreads (bps)
            attribution_date: Date for attribution (default today)

        Returns:
            AttributionResult with full decomposition
        """
        if attribution_date is None:
            attribution_date = date.today()

        # 1. CARRY P&L
        carry_pnl = PnLAttribution._compute_carry(cashflows_list, notionals)

        # 2. RATE P&L (reprice with today's curve, yesterday's spread)
        today_prices_same_spread = price_batch(cashflows_list, today_curve, yesterday_spreads)
        rate_pnl_values = yesterday_prices - today_prices_same_spread
        rate_pnl = np.sum(rate_pnl_values * np.array(notionals) / 100.0)

        # 3. SPREAD P&L (reprice with today's spread, today's curve)
        today_prices_today_spread = price_batch(cashflows_list, today_curve, today_spreads)
        spread_pnl_values = today_prices_same_spread - today_prices_today_spread
        spread_pnl = np.sum(spread_pnl_values * np.array(notionals) / 100.0)

        # 4. TOTAL P&L (full repricing)
        total_pnl = np.sum(
            (today_prices_today_spread - yesterday_prices) * np.array(notionals) / 100.0
        )

        # 5. RESIDUAL
        residual_pnl = total_pnl - carry_pnl - rate_pnl - spread_pnl

        # Explained percentage
        explained = carry_pnl + rate_pnl + spread_pnl
        explain_pct = abs(explained) / abs(total_pnl) if total_pnl != 0 else 0.0

        # Per-bond attribution
        per_bond = {}
        for i, cf in enumerate(cashflows_list):
            bond_carry = (cf.face_value * cf.coupon_rate * notionals[i]) / 365.0
            bond_rate = rate_pnl_values[i] * notionals[i] / 100.0
            bond_spread = spread_pnl_values[i] * notionals[i] / 100.0
            bond_total = (today_prices_today_spread[i] - yesterday_prices[i]) * notionals[i] / 100.0
            bond_residual = bond_total - bond_carry - bond_rate - bond_spread

            per_bond[cf.bond_id] = BondAttributionResult(
                bond_id=cf.bond_id,
                notional=notionals[i],
                carry_pnl=bond_carry,
                rate_pnl=bond_rate,
                spread_pnl=bond_spread,
                residual_pnl=bond_residual,
                total_pnl=bond_total,
                explained_pct=(bond_carry + bond_rate + bond_spread) / bond_total
                if bond_total != 0
                else 0.0,
            )

        result = AttributionResult(
            date=attribution_date,
            total_pnl=total_pnl,
            carry_pnl=carry_pnl,
            rate_pnl=rate_pnl,
            spread_pnl=spread_pnl,
            residual_pnl=residual_pnl,
            per_bond=per_bond,
            explain_pct=explain_pct,
        )

        logger.info(f"Daily attribution: Total={total_pnl:.2f}, Explained={explain_pct:.1%}")

        return result

    @staticmethod
    def _compute_carry(
        cashflows_list: list[CashFlowSchedule],
        notionals: list[float],
    ) -> float:
        """
        Compute daily carry: coupon earned each day.

        carry = sum(face_value * coupon_rate * notional) / 365

        Args:
            cashflows_list: Bond cash flows
            notionals: Notional amounts

        Returns:
            Daily carry P&L in currency units
        """
        total_carry = 0.0

        for cf, notional in zip(cashflows_list, notionals):
            daily_carry = (cf.face_value * cf.coupon_rate * notional) / 365.0
            total_carry += daily_carry

        return total_carry

    @staticmethod
    def attribution_time_series(
        daily_attributions: list[AttributionResult],
    ) -> pd.DataFrame:
        """
        Convert list of daily attributions to time series DataFrame.

        Args:
            daily_attributions: List of AttributionResult objects

        Returns:
            DataFrame with columns: [date, carry, rate, spread, residual, total]
        """
        records = []

        for attr in daily_attributions:
            records.append({
                "date": attr.date,
                "carry": attr.carry_pnl,
                "rate": attr.rate_pnl,
                "spread": attr.spread_pnl,
                "residual": attr.residual_pnl,
                "total": attr.total_pnl,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        return df

    @staticmethod
    def validate_attribution(attr: AttributionResult, tolerance: float = 1.0) -> bool:
        """
        Validate that attribution components sum to total within tolerance.

        Formula: carry + rate + spread + residual = total (within tolerance Rs)

        Args:
            attr: Attribution result
            tolerance: Maximum error in currency units (default Rs 1)

        Returns:
            True if valid, False otherwise
        """
        computed_sum = attr.carry_pnl + attr.rate_pnl + attr.spread_pnl + attr.residual_pnl
        error = abs(computed_sum - attr.total_pnl)

        is_valid = error <= tolerance

        if not is_valid:
            logger.warning(f"Attribution validation failed: error={error:.2f} > tolerance={tolerance}")

        return is_valid
