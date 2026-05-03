"""
Key Rate DV01 (KR01) computation.

KR01 shows how much portfolio value changes per basis point move at each tenor bucket.
This is how traders actually manage rate risk - not DV01 (which only tells you total),
but KR01 (which tells you WHERE the risk lives on the curve).

Implementation:
For each key rate tenor t_k, apply a triangular bump (1bp at t_k, fading to neighbors),
reprice the bond, and compute the value change.
"""

import logging
from dataclasses import dataclass

import numpy as np

from config.settings import settings
from curves.bootstrapper import SpotCurve
from pricing.bond_pricer import BondPricer
from pricing.cashflow_generator import CashFlowSchedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KR01Result:
    """KR01 computation result."""

    bond_id: str
    kr01_by_tenor: dict[float, float]  # {tenor: kr01_value}
    total_kr01: float
    notional: float


class KR01Calculator:
    """Computes Key Rate DV01 at standard tenor buckets."""

    KEY_RATE_TENORS = settings.KEY_RATE_TENORS  # [0.5, 1, 2, 3, 5, 7, 10, 20, 30]

    @staticmethod
    def compute_kr01(
        cashflows: CashFlowSchedule,
        spot_curve: SpotCurve,
        notional: float,
        spread_bps: float = 0.0,
        bump_bps: float = 1.0,
    ) -> KR01Result:
        """
        Compute Key Rate DV01 for a bond.

        Algorithm:
        For each key rate tenor t_k:
            1. Create bumped curve: +bump_bps at t_k, linearly fading to neighbors
            2. Reprice bond on bumped curve
            3. KR01_k = |Price_original - Price_bumped| * (notional/100)

        Args:
            cashflows: Bond cash flow schedule
            spot_curve: Current spot yield curve
            notional: Notional amount (e.g., Rs 1 crore)
            spread_bps: Credit spread in basis points
            bump_bps: Bump size (default 1 bp)

        Returns:
            KR01Result with KR01 at each tenor
        """
        # Original price
        price_original = BondPricer.price(cashflows, spot_curve, spread_bps)

        kr01_dict = {}

        for kr_tenor in KR01Calculator.KEY_RATE_TENORS:
            # Create triangular bump at this tenor
            bumped_curve = KR01Calculator._create_triangular_bump_curve(
                spot_curve, kr_tenor, bump_bps
            )

            # Reprice on bumped curve
            price_bumped = BondPricer.price(cashflows, bumped_curve, spread_bps)

            # KR01: price change in rupees
            price_change = price_original - price_bumped
            kr01_value = abs(price_change) * (notional / 100.0)

            kr01_dict[kr_tenor] = kr01_value
            logger.debug(f"KR01 at {kr_tenor}Y: {kr01_value:.2f} (price change: {price_change:.6f})")

        total_kr01 = sum(kr01_dict.values())

        logger.info(f"Total KR01: {total_kr01:.2f} across {len(kr01_dict)} tenor buckets")

        return KR01Result(
            bond_id=cashflows.bond_id,
            kr01_by_tenor=kr01_dict,
            total_kr01=total_kr01,
            notional=notional,
        )

    @staticmethod
    def _create_triangular_bump_curve(
        spot_curve: SpotCurve,
        bump_tenor: float,
        bump_bps: float = 1.0,
    ) -> SpotCurve:
        """
        Create a bumped spot curve with triangular shock at bump_tenor.

        The triangular bump:
        - Full bump_bps at bump_tenor
        - Linearly fades to 0 at neighboring key rate tenors
        - 0 everywhere else

        Algorithm:
        For each tenor t in the original curve:
            1. Find neighboring key rate tenors (if any)
            2. Compute linear interpolation of shock between neighbors
            3. Add shock to original spot rate

        Args:
            spot_curve: Original spot curve
            bump_tenor: Tenor at which to apply full bump
            bump_bps: Bump size in basis points

        Returns:
            New SpotCurve with bumped rates
        """
        bump_decimal = bump_bps / 10000.0  # Convert bps to decimal

        # Find neighboring key rate tenors
        kr_tenors = sorted(KR01Calculator.KEY_RATE_TENORS)

        if bump_tenor not in kr_tenors:
            raise ValueError(f"bump_tenor {bump_tenor} not in KEY_RATE_TENORS")

        idx = kr_tenors.index(bump_tenor)
        t_lower = kr_tenors[idx - 1] if idx > 0 else None
        t_upper = kr_tenors[idx + 1] if idx < len(kr_tenors) - 1 else None

        # Create bumped rates
        bumped_rates = spot_curve.spot_rates.copy()

        for i, tenor in enumerate(spot_curve.tenors):
            # Determine shock at this tenor
            if tenor == bump_tenor:
                shock = bump_decimal
            elif t_lower is not None and tenor >= t_lower and tenor < bump_tenor:
                # Between lower neighbor and bump tenor: linear interpolation
                fraction = (tenor - t_lower) / (bump_tenor - t_lower)
                shock = bump_decimal * fraction
            elif t_upper is not None and tenor > bump_tenor and tenor <= t_upper:
                # Between bump tenor and upper neighbor: linear interpolation
                fraction = (t_upper - tenor) / (t_upper - bump_tenor)
                shock = bump_decimal * fraction
            else:
                # Outside neighborhood: no shock
                shock = 0.0

            bumped_rates[i] = spot_curve.spot_rates[i] + shock

        # Create new SpotCurve with bumped rates
        from curves.bootstrapper import SpotCurve
        bumped_curve = SpotCurve(
            tenors=spot_curve.tenors,
            spot_rates=bumped_rates,
            as_of_date=spot_curve.as_of_date,
        )

        return bumped_curve

    @staticmethod
    def portfolio_kr01(
        cashflows_list: list[CashFlowSchedule],
        spot_curve: SpotCurve,
        notionals: list[float],
        spreads_bps: list[float] | float = 0.0,
    ) -> dict[float, float]:
        """
        Compute portfolio KR01: sum of individual bond KR01s.

        Args:
            cashflows_list: List of bond cash flow schedules
            spot_curve: Spot curve
            notionals: Notional amounts (one per bond)
            spreads_bps: Spreads (single value or list)

        Returns:
            Dict mapping tenor to total KR01
        """
        if isinstance(spreads_bps, (int, float)):
            spreads = [spreads_bps] * len(cashflows_list)
        else:
            spreads = spreads_bps

        portfolio_kr01 = {tenor: 0.0 for tenor in KR01Calculator.KEY_RATE_TENORS}

        for cf, notional, spread in zip(cashflows_list, notionals, spreads):
            kr01_result = KR01Calculator.compute_kr01(cf, spot_curve, notional, spread)

            for tenor, kr01_value in kr01_result.kr01_by_tenor.items():
                portfolio_kr01[tenor] += kr01_value

        logger.info(f"Portfolio KR01 across {len(portfolio_kr01)} tenors: {sum(portfolio_kr01.values()):.2f}")

        return portfolio_kr01

    @staticmethod
    def kr01_percentage_by_tenor(kr01_dict: dict[float, float]) -> dict[float, float]:
        """
        Compute KR01 as percentage of total.

        Args:
            kr01_dict: Dict from compute_kr01 or portfolio_kr01

        Returns:
            Dict with KR01 percentages (sum to 100%)
        """
        total = sum(kr01_dict.values())

        if total == 0:
            return {tenor: 0.0 for tenor in kr01_dict.keys()}

        return {tenor: (kr01 / total) * 100.0 for tenor, kr01 in kr01_dict.items()}

    @staticmethod
    def kr01_concentration(
        kr01_dict: dict[float, float],
        threshold_pct: float = 0.60,
    ) -> list[float]:
        """
        Find tenor buckets with high concentration of KR01.

        Args:
            kr01_dict: KR01 by tenor
            threshold_pct: Concentration threshold (e.g., 0.60 for 60%)

        Returns:
            List of tenors with KR01 > threshold
        """
        pct_dict = KR01Calculator.kr01_percentage_by_tenor(kr01_dict)

        high_concentration = [tenor for tenor, pct in pct_dict.items() if pct > threshold_pct * 100]

        return high_concentration
