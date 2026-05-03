"""
Real-time alert monitoring for portfolio risk and performance.

Checks VaR limits, concentration, duration drift, spread changes, KR01 concentration.
All checks are wrapped in try/except to prevent dashboard crashes.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    """A single alert."""

    alert_id: str
    severity: str  # "critical" | "warning" | "info"
    metric: str  # e.g., "VaR Limit", "Concentration"
    current_value: float
    threshold: float
    breach_pct: float  # (current - threshold) / threshold * 100
    message: str
    remediation: str
    timestamp: datetime

    @property
    def severity_rank(self) -> int:
        """Return severity rank for sorting (higher = more severe)."""
        return {"critical": 3, "warning": 2, "info": 1}.get(self.severity, 0)


class AlertMonitor:
    """Monitors portfolio for alerts and risk breaches."""

    @staticmethod
    def check_var_limit(
        var_95: float,
        portfolio_value: float,
        var_limit_bps: float | None = None,
    ) -> Alert | None:
        """
        Check if VaR breaches limit.

        Args:
            var_95: VaR at 95% confidence (in currency units)
            portfolio_value: Current portfolio value (in currency units)
            var_limit_bps: VaR limit as % of portfolio (default from config)

        Returns:
            Alert if limit breached, None otherwise
        """
        try:
            if var_limit_bps is None:
                var_limit_bps = settings.VAR_LIMIT_BPS if hasattr(settings, "VAR_LIMIT_BPS") else 150

            # Convert % to absolute amount
            var_limit_abs = portfolio_value * (var_limit_bps / 10000.0)

            if var_95 > var_limit_abs:
                breach_pct = (var_95 - var_limit_abs) / var_limit_abs * 100
                severity = "critical" if breach_pct > 50 else "warning"

                return Alert(
                    alert_id=str(uuid.uuid4()),
                    severity=severity,
                    metric="VaR 95% Limit",
                    current_value=var_95,
                    threshold=var_limit_abs,
                    breach_pct=breach_pct,
                    message=(
                        f"VaR 95% is ${var_95/1e6:.2f}M, "
                        f"exceeds limit of ${var_limit_abs/1e6:.2f}M ({var_limit_bps}bps)"
                    ),
                    remediation="Reduce long-duration positions or increase hedges",
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.warning(f"Error checking VaR limit: {e}")
            return None

    @staticmethod
    def check_duration_drift(
        portfolio_duration: float,
        target_duration: float,
        tolerance_years: float = 1.0,
    ) -> Alert | None:
        """
        Check if portfolio duration drifts from target.

        Args:
            portfolio_duration: Current portfolio duration (years)
            target_duration: Target duration (years)
            tolerance_years: Tolerance band (default 1.0 year)

        Returns:
            Alert if drift exceeds tolerance, None otherwise
        """
        try:
            drift = abs(portfolio_duration - target_duration)

            if drift > tolerance_years:
                severity = "critical" if drift > tolerance_years * 2 else "warning"
                breach_pct = (drift / tolerance_years) * 100

                return Alert(
                    alert_id=str(uuid.uuid4()),
                    severity=severity,
                    metric="Duration Drift",
                    current_value=portfolio_duration,
                    threshold=target_duration,
                    breach_pct=breach_pct,
                    message=(
                        f"Portfolio duration {portfolio_duration:.2f}Y "
                        f"drifted {drift:.2f}Y from target {target_duration:.2f}Y"
                    ),
                    remediation="Rebalance portfolio toward target duration",
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.warning(f"Error checking duration drift: {e}")
            return None

    @staticmethod
    def check_concentration(
        notionals: list[float],
        bond_names: list[str],
        portfolio_value: float,
        max_single_pct: float = 0.25,
        sectors: list[str] | None = None,
    ) -> list[Alert]:
        """
        Check for concentration risk: single position or sector too large.

        Args:
            notionals: Bond notional amounts
            bond_names: Bond names/IDs
            portfolio_value: Total portfolio value
            max_single_pct: Max single position as % (default 25%)
            sectors: Sector for each bond (optional)

        Returns:
            List of alerts (one per position exceeding limit)
        """
        alerts = []

        try:
            # Check single position concentration
            for notional, bond_name in zip(notionals, bond_names):
                position_pct = (notional / portfolio_value) * 100
                max_abs = portfolio_value * (max_single_pct / 100.0)

                if notional > max_abs:
                    breach_pct = (notional - max_abs) / max_abs * 100
                    severity = "critical" if position_pct > max_single_pct * 2 else "warning"

                    alerts.append(Alert(
                        alert_id=str(uuid.uuid4()),
                        severity=severity,
                        metric="Position Concentration",
                        current_value=notional,
                        threshold=max_abs,
                        breach_pct=breach_pct,
                        message=(
                            f"{bond_name} position is {position_pct:.1f}% of portfolio, "
                            f"exceeds {max_single_pct*100:.1f}% limit"
                        ),
                        remediation=f"Reduce {bond_name} position or diversify",
                        timestamp=datetime.now(),
                    ))

            # Check sector concentration if provided
            if sectors and len(sectors) == len(notionals):
                sector_totals = {}
                for notional, sector in zip(notionals, sectors):
                    sector_totals[sector] = sector_totals.get(sector, 0) + notional

                sector_limit = portfolio_value * (max_single_pct * 3 / 100.0)  # Allow 3x for sector

                for sector, total in sector_totals.items():
                    if total > sector_limit:
                        sector_pct = (total / portfolio_value) * 100
                        alerts.append(Alert(
                            alert_id=str(uuid.uuid4()),
                            severity="warning",
                            metric="Sector Concentration",
                            current_value=total,
                            threshold=sector_limit,
                            breach_pct=((total - sector_limit) / sector_limit) * 100,
                            message=f"{sector} sector is {sector_pct:.1f}% of portfolio",
                            remediation=f"Rebalance away from {sector} sector",
                            timestamp=datetime.now(),
                        ))

            return alerts

        except Exception as e:
            logger.warning(f"Error checking concentration: {e}")
            return []

    @staticmethod
    def check_spread_widening(
        current_spreads_bps: list[float],
        previous_spreads_bps: list[float],
        bond_ids: list[str],
        alert_threshold_bps: float | None = None,
    ) -> list[Alert]:
        """
        Check for significant spread widening.

        Args:
            current_spreads_bps: Current spreads (bps)
            previous_spreads_bps: Previous spreads (bps)
            bond_ids: Bond IDs
            alert_threshold_bps: Alert if widening > this (default 25bps)

        Returns:
            List of alerts for bonds with significant widening
        """
        alerts = []

        try:
            if alert_threshold_bps is None:
                alert_threshold_bps = 25.0

            for current, previous, bond_id in zip(current_spreads_bps, previous_spreads_bps, bond_ids):
                spread_change = current - previous

                if spread_change > alert_threshold_bps:
                    severity = "critical" if spread_change > alert_threshold_bps * 2 else "warning"

                    alerts.append(Alert(
                        alert_id=str(uuid.uuid4()),
                        severity=severity,
                        metric="Spread Widening",
                        current_value=current,
                        threshold=previous,
                        breach_pct=(spread_change / previous) * 100 if previous != 0 else 0,
                        message=(
                            f"{bond_id} spread widened {spread_change:.1f}bps "
                            f"(from {previous:.1f} to {current:.1f})"
                        ),
                        remediation=f"Monitor credit of {bond_id}, consider reducing position",
                        timestamp=datetime.now(),
                    ))

            return alerts

        except Exception as e:
            logger.warning(f"Error checking spread widening: {e}")
            return []

    @staticmethod
    def check_kr01_concentration(
        kr01_by_tenor: dict[float, float],
        threshold_pct: float | None = None,
    ) -> Alert | None:
        """
        Check for concentration in KR01 at specific tenor.

        Args:
            kr01_by_tenor: KR01 values by tenor
            threshold_pct: Alert if single tenor > this % of total KR01 (default 40%)

        Returns:
            Alert if concentration high, None otherwise
        """
        try:
            if threshold_pct is None:
                threshold_pct = 40.0

            if not kr01_by_tenor:
                return None

            total_kr01 = sum(kr01_by_tenor.values())
            if total_kr01 == 0:
                return None

            # Find tenor with highest KR01
            max_tenor = max(kr01_by_tenor.items(), key=lambda x: x[1])
            max_pct = (max_tenor[1] / total_kr01) * 100

            if max_pct > threshold_pct:
                severity = "warning" if max_pct > threshold_pct else "info"

                return Alert(
                    alert_id=str(uuid.uuid4()),
                    severity=severity,
                    metric="KR01 Concentration",
                    current_value=max_pct,
                    threshold=threshold_pct,
                    breach_pct=((max_pct - threshold_pct) / threshold_pct) * 100,
                    message=(
                        f"KR01 concentration at {max_tenor[0]:.1f}Y tenor: "
                        f"{max_pct:.1f}% of total KR01"
                    ),
                    remediation="Diversify tenor exposure across curve",
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.warning(f"Error checking KR01 concentration: {e}")
            return None

    @classmethod
    def check_all(
        cls,
        var_95: float,
        portfolio_value: float,
        notionals: list[float],
        bond_names: list[str],
        bond_ids: list[str],
        current_spreads_bps: list[float],
        previous_spreads_bps: list[float] | None = None,
        portfolio_duration: float | None = None,
        target_duration: float = 5.0,
        kr01_by_tenor: dict[float, float] | None = None,
        sectors: list[str] | None = None,
    ) -> list[Alert]:
        """
        Run all alert checks and return sorted list.

        Args:
            var_95: VaR at 95% confidence
            portfolio_value: Current portfolio value
            notionals: Bond notional amounts
            bond_names: Bond names/IDs
            bond_ids: Bond IDs
            current_spreads_bps: Current spreads
            previous_spreads_bps: Previous spreads (optional, for widening check)
            portfolio_duration: Current portfolio duration (optional)
            target_duration: Target duration (default 5.0)
            kr01_by_tenor: KR01 by tenor (optional)
            sectors: Bond sectors (optional)

        Returns:
            Sorted list of alerts (critical > warning > info)
        """
        alerts = []

        # VaR limit
        var_alert = cls.check_var_limit(var_95, portfolio_value)
        if var_alert:
            alerts.append(var_alert)

        # Duration drift
        if portfolio_duration is not None:
            duration_alert = cls.check_duration_drift(portfolio_duration, target_duration)
            if duration_alert:
                alerts.append(duration_alert)

        # Concentration
        conc_alerts = cls.check_concentration(notionals, bond_names, portfolio_value, sectors=sectors)
        alerts.extend(conc_alerts)

        # Spread widening
        if previous_spreads_bps is not None:
            spread_alerts = cls.check_spread_widening(current_spreads_bps, previous_spreads_bps, bond_ids)
            alerts.extend(spread_alerts)

        # KR01 concentration
        if kr01_by_tenor is not None:
            kr01_alert = cls.check_kr01_concentration(kr01_by_tenor)
            if kr01_alert:
                alerts.append(kr01_alert)

        # Sort by severity
        alerts.sort(key=lambda a: (-a.severity_rank, a.timestamp))

        logger.info(f"Alert check complete: {len(alerts)} alerts")

        return alerts
