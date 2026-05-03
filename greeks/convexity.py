"""Bond convexity calculation."""

import logging

from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import BondPricer
from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


class ConvexityCalculator:
    """Computes bond convexity."""

    @staticmethod
    def convexity_from_ytm(
        cashflows: CashFlowSchedule,
        ytm: float,
    ) -> float:
        """
        Compute Macaulay convexity from YTM.

        Convexity = (1/Price) * sum(t_i * (t_i+1) * CF_i / (1+y)^(t_i+2))

        Simplification of the full formula.

        Args:
            cashflows: Bond cash flows
            ytm: Yield to maturity

        Returns:
            Convexity
        """
        price = sum(cf.amount / ((1.0 + ytm) ** cf.tenor) for cf in cashflows.cash_flows)

        if price == 0:
            return 0.0

        numerator = 0.0
        for cf in cashflows.cash_flows:
            numerator += cf.tenor * (cf.tenor + 1) * cf.amount / ((1.0 + ytm) ** (cf.tenor + 2))

        convexity = numerator / (price * ((1.0 + ytm / cashflows.coupon_frequency) ** 2))

        return convexity

    @staticmethod
    def convexity_numerical(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        spread_bps: float = 0.0,
        bump_bps: float = 1.0,
    ) -> float:
        """
        Compute numerical convexity via finite difference.

        Conv = (P_up + P_down - 2*P_0) / (P_0 * dy^2)

        where dy = bump_bps / 10000

        Args:
            cashflows: Bond cash flows
            spot_curve: Spot curve
            spread_bps: Credit spread
            bump_bps: Bump size (default 1 bp)

        Returns:
            Convexity
        """
        # Unfortunately, can't easily bump the immutable SpotCurve
        # As workaround, use spread bumping as proxy

        p0 = BondPricer.price(cashflows, spot_curve, spread_bps)
        p_up = BondPricer.price(cashflows, spot_curve, spread_bps + bump_bps)
        p_down = BondPricer.price(cashflows, spot_curve, spread_bps - bump_bps)

        if p0 == 0:
            return 0.0

        dy = bump_bps / 10000.0
        dy_squared = dy * dy

        convexity = (p_up + p_down - 2 * p0) / (p0 * dy_squared)

        return convexity

    @staticmethod
    def price_change_with_convexity(
        mod_duration: float,
        convexity: float,
        dy_bps: float,
    ) -> float:
        """
        Estimate price change using duration and convexity.

        dP/P = -ModD * dy + 0.5 * Conv * dy^2

        where dy = dy_bps / 10000

        Args:
            mod_duration: Modified duration
            convexity: Convexity
            dy_bps: Yield change in basis points

        Returns:
            Percentage price change
        """
        dy = dy_bps / 10000.0

        duration_effect = -mod_duration * dy
        convexity_effect = 0.5 * convexity * (dy * dy)

        total_change = duration_effect + convexity_effect

        return total_change

    @staticmethod
    def dollar_convexity(
        cashflows: CashFlowSchedule,
        ytm: float,
        notional: float,
    ) -> float:
        """
        Compute convexity in dollar (currency) terms.

        Dollar Convexity = Convexity * Price * Notional / 100^2

        Args:
            cashflows: Bond cash flows
            ytm: Yield to maturity
            notional: Notional amount

        Returns:
            Dollar convexity
        """
        from pricing.ytm_solver import bond_price_from_ytm

        price = bond_price_from_ytm(cashflows, ytm)
        convexity = ConvexityCalculator.convexity_from_ytm(cashflows, ytm)

        dollar_convexity = convexity * (price / 100.0) * (notional / 10000.0)

        return dollar_convexity
