"""
Document Validation Module
Checks for missing documents, conflicting values, and suspicious information.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

REQUIRED_DOCUMENTS = [
    "Term Sheet",
    "Loan Application",
    "Purchase Contract",
    "Appraisal",
    "Scope of Work",
    "Title Commitment",
    "Insurance Binder",
    "Credit Report",
    "Bank Statement",
]

RECOMMENDED_DOCUMENTS = [
    "Flood Certificate",
    "Operating Agreement",
    "Certificate of Good Standing",
    "Government ID",
    "Borrower Track Record",
    "Background Check",
]


@dataclass
class ValidationIssue:
    severity: str    # "error", "warning", "info"
    code: str        # Short identifier
    description: str
    affected_documents: list[str] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class ValidationResult:
    missing_required: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    overall_completeness_pct: float = 0.0

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == "error" for i in self.issues) or bool(self.missing_required)


def validate_documents(
    classified_categories: list[str],
    extraction_result,
    financial_metrics,
) -> ValidationResult:
    """
    Run all validation checks.
    """
    result = ValidationResult()

    # 1. Check for missing required documents
    found_cats = set(classified_categories)
    result.missing_required = [d for d in REQUIRED_DOCUMENTS if d not in found_cats]
    result.missing_recommended = [d for d in RECOMMENDED_DOCUMENTS if d not in found_cats]

    for missing in result.missing_required:
        result.issues.append(ValidationIssue(
            severity="error",
            code="MISSING_REQUIRED_DOC",
            description=f"Required document missing: {missing}",
            suggested_action=f"Obtain and upload a {missing} document",
        ))

    for missing in result.missing_recommended:
        result.issues.append(ValidationIssue(
            severity="warning",
            code="MISSING_RECOMMENDED_DOC",
            description=f"Recommended document not found: {missing}",
            suggested_action=f"Consider obtaining {missing} for a more complete underwriting",
        ))

    # 2. Check for missing critical extracted values
    ext = extraction_result
    _check_missing_field(result, ext.loan.loan_amount, "Loan Amount", "MISSING_LOAN_AMOUNT")
    _check_missing_field(result, ext.loan.purchase_price, "Purchase Price", "MISSING_PURCHASE_PRICE")
    _check_missing_field(result, ext.loan.as_is_value, "As-Is Property Value", "MISSING_AS_IS_VALUE")
    _check_missing_field(result, ext.loan.arv, "After Repair Value (ARV)", "MISSING_ARV")
    _check_missing_field(result, ext.borrower.credit_score, "Credit Score", "MISSING_CREDIT_SCORE")
    _check_missing_field(result, ext.borrower.bank_balance, "Bank Balance / Liquidity", "MISSING_BANK_BALANCE")

    # 3. Financial metric consistency checks
    if financial_metrics:
        fm = financial_metrics

        # Cross-document value conflict checks
        _check_loan_vs_as_is(result, fm)
        _check_arv_sanity(result, fm)
        _check_equity_adequacy(result, fm)

    # 4. Credit score checks
    credit_val = ext.borrower.credit_score.value
    if credit_val and credit_val != "Not Found":
        try:
            score = float(credit_val)
            if score < 580:
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="LOW_CREDIT_SCORE",
                    description=f"Credit score {score:.0f} is below minimum threshold (580)",
                    suggested_action="Borrower does not meet minimum credit requirements",
                ))
            elif score < 640:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="BORDERLINE_CREDIT_SCORE",
                    description=f"Credit score {score:.0f} is below preferred threshold (640)",
                    suggested_action="Consider requesting additional compensating factors",
                ))
        except (ValueError, TypeError):
            pass

    # 5. Liquidity check
    balance_val = ext.borrower.bank_balance.value
    loan_val = ext.loan.loan_amount.value
    if balance_val and balance_val != "Not Found" and loan_val and loan_val != "Not Found":
        try:
            balance = float(balance_val)
            loan = float(loan_val)
            if balance < loan * 0.05:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="LOW_LIQUIDITY",
                    description=(
                        f"Borrower liquidity (${balance:,.0f}) is less than 5% of "
                        f"loan amount (${loan:,.0f})"
                    ),
                    suggested_action="Verify borrower has sufficient reserves",
                ))
        except (ValueError, TypeError):
            pass

    # Completeness score
    total_docs = len(REQUIRED_DOCUMENTS) + len(RECOMMENDED_DOCUMENTS)
    found_docs = len([d for d in REQUIRED_DOCUMENTS + RECOMMENDED_DOCUMENTS if d in found_cats])
    result.overall_completeness_pct = round((found_docs / total_docs) * 100, 1)

    return result


def _check_missing_field(result: ValidationResult, field, label: str, code: str):
    """Add warning if field value is 'Not Found'."""
    from src.extraction.information_extractor import NOT_FOUND
    if field.value == NOT_FOUND:
        result.issues.append(ValidationIssue(
            severity="warning",
            code=code,
            description=f"Could not extract: {label}",
            suggested_action=f"Verify {label} is present in uploaded documents",
        ))


def _check_loan_vs_as_is(result: ValidationResult, fm):
    """Check LTV thresholds."""
    if fm.as_is_ltv and fm.as_is_ltv > 85:
        result.issues.append(ValidationIssue(
            severity="error",
            code="EXCESSIVE_AS_IS_LTV",
            description=f"As-Is LTV of {fm.as_is_ltv:.1f}% exceeds maximum (85%)",
            suggested_action="Reduce loan amount or obtain a higher appraisal",
        ))


def _check_arv_sanity(result: ValidationResult, fm):
    """Check ARV vs as-is value."""
    if fm.arv and fm.as_is_value and fm.arv < fm.as_is_value:
        result.issues.append(ValidationIssue(
            severity="error",
            code="ARV_BELOW_AS_IS",
            description=(
                f"ARV (${fm.arv:,.0f}) is lower than As-Is Value (${fm.as_is_value:,.0f}). "
                "This is inconsistent."
            ),
            suggested_action="Verify ARV and As-Is Value figures in the appraisal",
        ))


def _check_equity_adequacy(result: ValidationResult, fm):
    """Check borrower has skin in the game."""
    if fm.borrower_equity is not None and fm.borrower_equity < 0:
        result.issues.append(ValidationIssue(
            severity="error",
            code="NEGATIVE_EQUITY",
            description="Loan amount exceeds total project cost — borrower has negative equity",
            suggested_action="Reduce loan amount",
        ))

