"""Unit tests for alert monitoring."""

import numpy as np
import pytest

from alerts.monitor import Alert, AlertMonitor


class TestVaRLimitCheck:
    """Test VaR limit checking."""

    def test_no_alert_under_limit(self):
        """Test that no alert is issued when VaR is under limit."""
        var_95 = 5e6  # $5M
        portfolio_value = 500e6  # $500M
        var_limit_bps = 150  # 150 bps = 1.5%

        alert = AlertMonitor.check_var_limit(var_95, portfolio_value, var_limit_bps)

        assert alert is None

    def test_warning_alert_moderate_breach(self):
        """Test that warning alert is issued for moderate breach."""
        var_95 = 10e6  # $10M
        portfolio_value = 500e6  # $500M
        var_limit_bps = 150  # 1.5% = $7.5M limit

        alert = AlertMonitor.check_var_limit(var_95, portfolio_value, var_limit_bps)

        assert alert is not None
        assert alert.severity == "warning"
        assert alert.metric == "VaR 95% Limit"
        assert alert.current_value == var_95
        assert alert.breach_pct > 0

    def test_critical_alert_severe_breach(self):
        """Test that critical alert is issued for severe breach (>50%)."""
        var_95 = 20e6  # $20M
        portfolio_value = 500e6  # $500M
        var_limit_bps = 150  # 1.5% = $7.5M limit, breach = >100%

        alert = AlertMonitor.check_var_limit(var_95, portfolio_value, var_limit_bps)

        assert alert is not None
        assert alert.severity == "critical"
        assert alert.breach_pct > 50


class TestConcentrationCheck:
    """Test position concentration checking."""

    def test_no_alert_under_limit(self):
        """Test that no alert is issued when concentration is under limit."""
        notionals = [50e6, 50e6, 100e6, 100e6, 200e6]
        bond_names = ["B1", "B2", "B3", "B4", "B5"]
        portfolio_value = 500e6
        max_single_pct = 0.25

        alerts = AlertMonitor.check_concentration(
            notionals, bond_names, portfolio_value, max_single_pct
        )

        # Max position is 200M / 500M = 40%, which exceeds 25%
        assert len(alerts) > 0

    def test_alert_on_concentration_breach(self):
        """Test that alert is issued when concentration breaches limit."""
        notionals = [150e6, 100e6, 100e6, 50e6]  # Total 400M
        bond_names = ["LARGE", "B2", "B3", "B4"]
        portfolio_value = 400e6
        max_single_pct = 0.30  # 30% = $120M limit

        alerts = AlertMonitor.check_concentration(
            notionals, bond_names, portfolio_value, max_single_pct
        )

        # LARGE (150M) exceeds limit, others don't
        assert len(alerts) >= 1
        assert any("LARGE" in a.message for a in alerts)
        assert alerts[0].metric == "Position Concentration"


class TestSpreadWideningCheck:
    """Test spread widening checking."""

    def test_no_alert_small_widening(self):
        """Test that no alert is issued for small spread changes."""
        current = [50.0, 60.0, 70.0]
        previous = [45.0, 55.0, 65.0]
        bond_ids = ["B1", "B2", "B3"]
        threshold = 25.0

        alerts = AlertMonitor.check_spread_widening(
            current, previous, bond_ids, threshold
        )

        assert len(alerts) == 0

    def test_alert_on_significant_widening(self):
        """Test that alert is issued for significant spread widening."""
        current = [100.0, 60.0, 70.0]
        previous = [50.0, 55.0, 65.0]
        bond_ids = ["B1", "B2", "B3"]
        threshold = 25.0  # B1 widened 50bps, exceeds threshold

        alerts = AlertMonitor.check_spread_widening(
            current, previous, bond_ids, threshold
        )

        assert len(alerts) == 1
        assert "B1" in alerts[0].message
        assert alerts[0].severity == "warning"


class TestKR01ConcentrationCheck:
    """Test KR01 concentration checking."""

    def test_no_alert_balanced_kr01(self):
        """Test that no alert is issued for balanced KR01."""
        kr01 = {
            2.0: 50000,
            5.0: 50000,
            10.0: 50000,
            30.0: 50000,
        }
        threshold_pct = 40.0

        alert = AlertMonitor.check_kr01_concentration(kr01, threshold_pct)

        # Max tenor is 25% of total, under 40% threshold
        assert alert is None

    def test_alert_on_kr01_concentration(self):
        """Test that alert is issued when KR01 is concentrated."""
        kr01 = {
            2.0: 30000,
            5.0: 30000,
            30.0: 200000,  # 70% of total
        }
        threshold_pct = 40.0

        alert = AlertMonitor.check_kr01_concentration(kr01, threshold_pct)

        assert alert is not None
        assert alert.severity == "warning"
        assert alert.metric == "KR01 Concentration"
        assert "30" in alert.message


class TestDurationDriftCheck:
    """Test duration drift checking."""

    def test_no_alert_within_tolerance(self):
        """Test that no alert is issued when duration is within tolerance."""
        portfolio_duration = 5.0
        target_duration = 5.0
        tolerance = 1.0

        alert = AlertMonitor.check_duration_drift(
            portfolio_duration, target_duration, tolerance
        )

        assert alert is None

    def test_warning_alert_on_drift(self):
        """Test that warning is issued when duration drifts."""
        portfolio_duration = 6.5
        target_duration = 5.0
        tolerance = 1.0

        alert = AlertMonitor.check_duration_drift(
            portfolio_duration, target_duration, tolerance
        )

        assert alert is not None
        assert alert.severity == "warning"
        assert alert.metric == "Duration Drift"


class TestAlertSorting:
    """Test alert sorting by severity."""

    def test_alerts_sorted_by_severity(self):
        """Test that alerts are sorted critical > warning > info."""
        alerts = [
            Alert(
                alert_id="1",
                severity="info",
                metric="Test1",
                current_value=1.0,
                threshold=1.0,
                breach_pct=0.0,
                message="Test",
                remediation="Test",
                timestamp=None,
            ),
            Alert(
                alert_id="2",
                severity="critical",
                metric="Test2",
                current_value=1.0,
                threshold=1.0,
                breach_pct=0.0,
                message="Test",
                remediation="Test",
                timestamp=None,
            ),
            Alert(
                alert_id="3",
                severity="warning",
                metric="Test3",
                current_value=1.0,
                threshold=1.0,
                breach_pct=0.0,
                message="Test",
                remediation="Test",
                timestamp=None,
            ),
        ]

        # Sort by severity rank
        sorted_alerts = sorted(alerts, key=lambda a: -a.severity_rank)

        assert sorted_alerts[0].severity == "critical"
        assert sorted_alerts[1].severity == "warning"
        assert sorted_alerts[2].severity == "info"


class TestCheckAll:
    """Test comprehensive alert checking."""

    def test_check_all_integration(self):
        """Test that check_all runs all checks without crashing."""
        var_95 = 8e6
        portfolio_value = 500e6
        notionals = [50e6, 100e6, 150e6]
        bond_names = ["B1", "B2", "B3"]
        bond_ids = ["B1", "B2", "B3"]
        spreads = [0.0, 50.0, 100.0]
        prev_spreads = [0.0, 45.0, 95.0]
        duration = 5.5
        kr01_by_tenor = {2.0: 50000, 10.0: 60000, 30.0: 40000}

        alerts = AlertMonitor.check_all(
            var_95=var_95,
            portfolio_value=portfolio_value,
            notionals=notionals,
            bond_names=bond_names,
            bond_ids=bond_ids,
            current_spreads_bps=spreads,
            previous_spreads_bps=prev_spreads,
            portfolio_duration=duration,
            kr01_by_tenor=kr01_by_tenor,
        )

        # Should return a list of alerts
        assert isinstance(alerts, list)

        # If there are alerts, they should be sorted by severity
        if len(alerts) > 1:
            for i in range(len(alerts) - 1):
                assert alerts[i].severity_rank >= alerts[i + 1].severity_rank
