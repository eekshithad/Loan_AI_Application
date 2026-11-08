"""
Financial Calculations Module
Pure Python deterministic calculations for LTV, ARV LTV, LTC, and equity.
No LLM involved in calculations.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FinancialMetrics:
    """Results of all financial calculations."""
    # Inputs
    loan_amount: Optional[float]
    purchase_price: Optional[float]
    as_is_value: Optional[float]
    arv: Optional[float]
    rehab_budget: Optional[float]

    # Calculated outputs
    as_is_ltv: Optional[float] = None           # %
    arv_ltv: Optional[float] = None             # %
    ltc: Optional[float] = None                 # %
    total_project_cost: Optional[float] = None  # $
    borrower_equity: Optional[float] = None     # $

    # Risk flags
    flags: list[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


def calculate_metrics(
    loan_amount: Optional[float],
    purchase_price: Optional[float],
    as_is_value: Optional[float],
    arv: Optional[float],
    rehab_budget: Optional[float],
) -> FinancialMetrics:
    """
    Calculate all financial metrics using deterministic Python math.
    Returns None for metrics that cannot be computed due to missing inputs.
    """
    metrics = FinancialMetrics(
        loan_amount=loan_amount,
        purchase_price=purchase_price,
        as_is_value=as_is_value,
        arv=arv,
        rehab_budget=rehab_budget,
    )

    # As-Is LTV = Loan Amount / As-Is Value × 100
    if loan_amount and as_is_value and as_is_value > 0:
        metrics.as_is_ltv = round((loan_amount / as_is_value) * 100, 2)

    # ARV LTV = Loan Amount / ARV × 100
    if loan_amount and arv and arv > 0:
        metrics.arv_ltv = round((loan_amount / arv) * 100, 2)

    # Total Project Cost = Purchase Price + Rehab Budget
    if purchase_price is not None and rehab_budget is not None:
        metrics.total_project_cost = purchase_price + rehab_budget
    elif purchase_price is not None:
        metrics.total_project_cost = purchase_price

    # LTC = Loan Amount / (Purchase Price + Rehab Budget) × 100
    if loan_amount and metrics.total_project_cost and metrics.total_project_cost > 0:
        metrics.ltc = round((loan_amount / metrics.total_project_cost) * 100, 2)

    # Borrower Equity = Total Project Cost − Loan Amount
    if metrics.total_project_cost is not None and loan_amount is not None:
        metrics.borrower_equity = metrics.total_project_cost - loan_amount

    # Risk flags
    _add_risk_flags(metrics)

    return metrics


def _add_risk_flags(metrics: FinancialMetrics) -> None:
    """Add risk flags based on calculated metrics."""
    flags = []

    if metrics.as_is_ltv is not None:
        if metrics.as_is_ltv > 80:
            flags.append(f"⚠️ HIGH As-Is LTV: {metrics.as_is_ltv:.1f}% (threshold: 80%)")
        elif metrics.as_is_ltv > 70:
            flags.append(f"⚡ ELEVATED As-Is LTV: {metrics.as_is_ltv:.1f}% (threshold: 70%)")

    if metrics.arv_ltv is not None:
        if metrics.arv_ltv > 75:
            flags.append(f"⚠️ HIGH ARV LTV: {metrics.arv_ltv:.1f}% (threshold: 75%)")
        elif metrics.arv_ltv > 65:
            flags.append(f"⚡ ELEVATED ARV LTV: {metrics.arv_ltv:.1f}% (threshold: 65%)")

    if metrics.ltc is not None:
        if metrics.ltc > 90:
            flags.append(f"⚠️ HIGH LTC: {metrics.ltc:.1f}% (threshold: 90%)")
        elif metrics.ltc > 80:
            flags.append(f"⚡ ELEVATED LTC: {metrics.ltc:.1f}% (threshold: 80%)")

    if metrics.borrower_equity is not None:
        if metrics.borrower_equity < 0:
            flags.append("🚨 NEGATIVE borrower equity — loan exceeds total project cost")
        elif metrics.borrower_equity < 10000:
            flags.append(f"⚠️ LOW borrower equity: ${metrics.borrower_equity:,.0f}")

    # Cross-check: ARV should be greater than as-is value
    if metrics.arv and metrics.as_is_value:
        if metrics.arv < metrics.as_is_value:
            flags.append("🚨 ARV is LESS THAN as-is value — data may be incorrect")
        elif metrics.arv > metrics.as_is_value * 2.5:
            flags.append("⚠️ ARV is more than 2.5× as-is value — verify renovation scope")

    # Rehab budget sanity
    if metrics.rehab_budget and metrics.as_is_value:
        ratio = metrics.rehab_budget / metrics.as_is_value
        if ratio > 1.0:
            flags.append(f"⚠️ Rehab budget ({ratio:.1%} of as-is value) is very high")

    metrics.flags = flags


def format_currency(value: Optional[float]) -> str:
    """Format a number as USD currency string."""
    if value is None:
        return "N/A"
    return f"${value:,.0f}"


def format_percent(value: Optional[float]) -> str:
    """Format a number as percentage string."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"

