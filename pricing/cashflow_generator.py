"""Bond cash flow generation and day count conventions."""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DayCountMethod(Enum):
    """Day count conventions for bond mathematics."""

    ACT_365 = "ACT/365"    # Actual days / 365 (Indian G-secs)
    ACT_ACT = "ACT/ACT"    # Actual days / actual days in year (US Treasuries)
    THIRTY_360 = "30/360"  # 30-day months, 360-day years


class CashFlow(NamedTuple):
    """Single bond cash flow."""

    date: date          # Payment date
    amount: float       # Amount in currency
    tenor: float        # Years from today to this cash flow
    cf_type: str        # "coupon" or "principal+coupon"


@dataclass
class CashFlowSchedule:
    """Complete cash flow schedule for a bond."""

    bond_id: str
    cash_flows: list[CashFlow]
    face_value: float
    coupon_rate: float  # Annual coupon rate
    coupon_frequency: int  # 1 (annual) or 2 (semi-annual)

    def total_cash_flows(self) -> float:
        """Sum of all cash flows."""
        return sum(cf.amount for cf in self.cash_flows)

    def principal_amount(self) -> float:
        """Total principal (usually face value)."""
        return self.face_value

    def maturity_date(self) -> date:
        """Maturity date of the bond."""
        return self.cash_flows[-1].date


def _days_between(d1: date, d2: date) -> int:
    """Count actual days between two dates."""
    return (d2 - d1).days


def _days_in_year(date_: date, method: DayCountMethod) -> int:
    """Get number of days in the year for a given date."""
    if method in (DayCountMethod.ACT_365, DayCountMethod.THIRTY_360):
        return 365
    elif method == DayCountMethod.ACT_ACT:
        # Check if leap year
        year = date_.year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 366
        else:
            return 365
    else:
        return 365


def _tenor_in_years(
    from_date: date,
    to_date: date,
    method: DayCountMethod = DayCountMethod.ACT_365,
) -> float:
    """
    Compute time between two dates in years using day count convention.

    Args:
        from_date: Start date
        to_date: End date
        method: Day count method

    Returns:
        Time in years as decimal
    """
    if method == DayCountMethod.ACT_365:
        actual_days = _days_between(from_date, to_date)
        return actual_days / 365.0

    elif method == DayCountMethod.ACT_ACT:
        # Simplified: treat as actual/actual 365 for consistency
        # Full ACT/ACT requires year-by-year calculation
        actual_days = _days_between(from_date, to_date)
        days_in_year = _days_in_year(from_date, method)
        return actual_days / days_in_year

    elif method == DayCountMethod.THIRTY_360:
        # 30/360: each month has 30 days, year has 360 days
        d1_day, d1_month, d1_year = from_date.day, from_date.month, from_date.year
        d2_day, d2_month, d2_year = to_date.day, to_date.month, to_date.year

        # Adjust for 30/360
        d1_day = min(d1_day, 30)
        d2_day = min(d2_day, 30) if d1_day == 30 else d2_day

        days = (360 * (d2_year - d1_year) + 30 * (d2_month - d1_month) + (d2_day - d1_day))
        return days / 360.0

    else:
        raise ValueError(f"Unknown day count method: {method}")


def generate_cashflow_schedule(
    bond_id: str,
    face_value: float,
    coupon_rate: float,
    coupon_frequency: int,
    maturity_date: date,
    issue_date: date,
    as_of_date: date | None = None,
    day_count_method: DayCountMethod = DayCountMethod.ACT_365,
) -> CashFlowSchedule:
    """
    Generate complete cash flow schedule for a bond.

    Args:
        bond_id: Bond identifier
        face_value: Face value (e.g., 100)
        coupon_rate: Annual coupon rate as decimal (e.g., 0.075 for 7.5%)
        coupon_frequency: 1 (annual) or 2 (semi-annual)
        maturity_date: Bond maturity date
        issue_date: Bond issue date
        as_of_date: Today's date (for computing tenors). Defaults to today.
        day_count_method: Day count convention

    Returns:
        CashFlowSchedule with all future cash flows

    Raises:
        ValueError: If inputs are invalid
    """
    if as_of_date is None:
        as_of_date = date.today()

    if maturity_date <= as_of_date:
        logger.warning(f"Bond {bond_id} has already matured (maturity: {maturity_date}, as_of: {as_of_date})")
        return CashFlowSchedule(bond_id=bond_id, cash_flows=[], face_value=face_value, coupon_rate=coupon_rate, coupon_frequency=coupon_frequency)

    if maturity_date <= issue_date:
        raise ValueError(f"Maturity date {maturity_date} must be after issue date {issue_date}")

    if coupon_frequency not in (1, 2, 4):
        raise ValueError(f"coupon_frequency must be 1, 2, or 4, got {coupon_frequency}")

    coupon_amount = (face_value * coupon_rate) / coupon_frequency

    # Generate coupon payment dates
    cash_flows = []

    # Determine first coupon date (usually 6 months or 12 months after issue)
    months_per_period = 12 // coupon_frequency
    new_month = issue_date.month + months_per_period
    new_year = issue_date.year
    if new_month > 12:
        new_year += (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1
    first_coupon = date(new_year, new_month, issue_date.day)

    current_coupon_date = first_coupon

    def _advance_coupon_date(d: date) -> date:
        """Advance coupon date by one period, handling month-end overflow."""
        nm = d.month + months_per_period
        ny = d.year
        while nm > 12:
            nm -= 12
            ny += 1
        try:
            return d.replace(year=ny, month=nm)
        except ValueError:
            if nm == 2:
                day = 28 if ny % 4 != 0 else 29
            elif nm in (4, 6, 9, 11):
                day = 30
            else:
                day = 31
            return date(ny, nm, min(d.day, day))

    # Generate all intermediate coupon dates (strictly before maturity)
    while current_coupon_date < maturity_date:
        if current_coupon_date > as_of_date:
            tenor = _tenor_in_years(as_of_date, current_coupon_date, method=day_count_method)
            cash_flows.append(
                CashFlow(date=current_coupon_date, amount=coupon_amount, tenor=tenor, cf_type="coupon")
            )
        current_coupon_date = _advance_coupon_date(current_coupon_date)

    # Always add final payment at maturity_date (principal + coupon)
    # This handles both: maturity landing exactly on a coupon date and broken-period bonds

    if maturity_date > as_of_date:
        maturity_tenor = _tenor_in_years(as_of_date, maturity_date, method=day_count_method)
        cash_flows.append(
            CashFlow(date=maturity_date, amount=coupon_amount + face_value, tenor=maturity_tenor, cf_type="principal+coupon")
        )

    if not cash_flows:
        logger.warning(f"No future cash flows for bond {bond_id}")

    schedule = CashFlowSchedule(
        bond_id=bond_id,
        cash_flows=cash_flows,
        face_value=face_value,
        coupon_rate=coupon_rate,
        coupon_frequency=coupon_frequency,
    )

    logger.debug(f"Generated {len(cash_flows)} cash flows for {bond_id}, matures {maturity_date}")
    return schedule
