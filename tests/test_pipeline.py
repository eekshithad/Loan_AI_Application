"""
Tests for financial calculations — the most critical deterministic module.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.validation.financial_calculator import calculate_metrics, format_currency, format_percent


class TestFinancialCalculations:

    def test_as_is_ltv_basic(self):
        """As-Is LTV = Loan / As-Is Value × 100"""
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=350000,
            arv=450000,
            rehab_budget=95000,
        )
        expected = round((285000 / 350000) * 100, 2)
        assert metrics.as_is_ltv == expected

    def test_arv_ltv_basic(self):
        """ARV LTV = Loan / ARV × 100"""
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=350000,
            arv=450000,
            rehab_budget=95000,
        )
        expected = round((285000 / 450000) * 100, 2)
        assert metrics.arv_ltv == expected

    def test_ltc_basic(self):
        """LTC = Loan / (Purchase + Rehab) × 100"""
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=350000,
            arv=450000,
            rehab_budget=95000,
        )
        expected = round((285000 / (220000 + 95000)) * 100, 2)
        assert metrics.ltc == expected

    def test_total_project_cost(self):
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=350000,
            arv=450000,
            rehab_budget=95000,
        )
        assert metrics.total_project_cost == 220000 + 95000

    def test_borrower_equity(self):
        """Equity = Total Project Cost - Loan Amount"""
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=350000,
            arv=450000,
            rehab_budget=95000,
        )
        expected = (220000 + 95000) - 285000
        assert metrics.borrower_equity == expected

    def test_missing_as_is_value_gives_none_ltv(self):
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=None,
            arv=450000,
            rehab_budget=95000,
        )
        assert metrics.as_is_ltv is None

    def test_missing_arv_gives_none_arv_ltv(self):
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=350000,
            arv=None,
            rehab_budget=95000,
        )
        assert metrics.arv_ltv is None

    def test_all_missing_gives_all_none(self):
        metrics = calculate_metrics(None, None, None, None, None)
        assert metrics.as_is_ltv is None
        assert metrics.arv_ltv is None
        assert metrics.ltc is None
        assert metrics.total_project_cost is None
        assert metrics.borrower_equity is None

    def test_high_ltv_flag(self):
        """LTV above 80% should trigger a risk flag."""
        metrics = calculate_metrics(
            loan_amount=850000,
            purchase_price=800000,
            as_is_value=1000000,
            arv=1200000,
            rehab_budget=100000,
        )
        # as_is_ltv = 85%
        flag_texts = " ".join(metrics.flags)
        assert "LTV" in flag_texts
        assert any("80%" in f or "HIGH" in f for f in metrics.flags)

    def test_negative_equity_flag(self):
        """Loan > total project cost should flag negative equity."""
        metrics = calculate_metrics(
            loan_amount=500000,
            purchase_price=200000,
            as_is_value=250000,
            arv=350000,
            rehab_budget=100000,
        )
        # Borrower equity = 300000 - 500000 = -200000
        assert metrics.borrower_equity == -200000
        assert any("NEGATIVE" in f.upper() for f in metrics.flags)

    def test_arv_below_as_is_flag(self):
        """ARV < as-is value should flag data inconsistency."""
        metrics = calculate_metrics(
            loan_amount=200000,
            purchase_price=250000,
            as_is_value=300000,
            arv=200000,  # ARV < as-is — data error
            rehab_budget=50000,
        )
        assert any("ARV" in f.upper() and "LESS" in f.upper() for f in metrics.flags)

    def test_format_currency(self):
        assert format_currency(285000) == "$285,000"
        assert format_currency(None) == "N/A"
        assert format_currency(1234567.89) == "$1,234,568"

    def test_format_percent(self):
        assert format_percent(81.25) == "81.2%"
        assert format_percent(None) == "N/A"
        assert format_percent(100.0) == "100.0%"

    def test_exact_values(self):
        """Verify exact sample case from the application."""
        # Sample from synthetic document set
        # Loan: $285,000, Purchase: $220,000, As-Is: $235,000, ARV: $375,000, Rehab: $95,000
        metrics = calculate_metrics(
            loan_amount=285000,
            purchase_price=220000,
            as_is_value=235000,
            arv=375000,
            rehab_budget=95000,
        )
        assert metrics.as_is_ltv == round(285000 / 235000 * 100, 2)
        assert metrics.arv_ltv == round(285000 / 375000 * 100, 2)
        assert metrics.ltc == round(285000 / 315000 * 100, 2)
        assert metrics.total_project_cost == 315000
        assert metrics.borrower_equity == 315000 - 285000


class TestDecisionModel:

    def test_model_loads_or_trains(self):
        from src.decision.loan_decision_model import LoanDecisionModel
        model = LoanDecisionModel()
        model.load_or_train()
        assert model.is_trained

    def test_prediction_returns_valid_decision(self):
        from src.decision.loan_decision_model import LoanDecisionModel, build_feature_vector
        model = LoanDecisionModel()
        model.load_or_train()
        vec = build_feature_vector(740, 65.0, 60.0, 72.0, 72500, 285000, 90.0, 5, 0)
        pred = model.predict(vec)
        assert pred["decision"] in ["APPROVE", "CONDITIONAL_APPROVAL", "DECLINE"]
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_strong_borrower_leans_approve(self):
        """High credit score, low LTV, good docs → should lean APPROVE."""
        from src.decision.loan_decision_model import LoanDecisionModel, build_feature_vector
        model = LoanDecisionModel()
        model.load_or_train()
        vec = build_feature_vector(780, 55.0, 48.0, 65.0, 100000, 200000, 100.0, 10, 0)
        pred = model.predict(vec)
        # With strong features, model should not decline
        assert pred["decision"] in ["APPROVE", "CONDITIONAL_APPROVAL"]

    def test_poor_borrower_leans_decline(self):
        """Low credit, very high LTV, missing docs → should lean DECLINE."""
        from src.decision.loan_decision_model import LoanDecisionModel, build_feature_vector
        model = LoanDecisionModel()
        model.load_or_train()
        vec = build_feature_vector(540, 95.0, 90.0, 105.0, 1000, 500000, 20.0, 0, 5)
        pred = model.predict(vec)
        assert pred["decision"] in ["DECLINE", "CONDITIONAL_APPROVAL"]


class TestClassification:

    def test_keyword_term_sheet(self):
        from src.classification.document_classifier import classify_document_keyword
        text = "TERM SHEET bridge loan interest rate 11.5% origination fee maturity date ARV"
        result = classify_document_keyword(text, "term_sheet.pdf")
        assert result.category == "Term Sheet"

    def test_keyword_appraisal(self):
        from src.classification.document_classifier import classify_document_keyword
        text = "APPRAISAL REPORT as-is value after repair value ARV comparable sales market value"
        result = classify_document_keyword(text, "appraisal.pdf")
        assert result.category == "Appraisal"

    def test_keyword_credit_report(self):
        from src.classification.document_classifier import classify_document_keyword
        text = "credit report FICO score 718 Equifax Experian TransUnion payment history"
        result = classify_document_keyword(text, "credit.pdf")
        assert result.category == "Credit Report"

    def test_empty_text_gives_other(self):
        from src.classification.document_classifier import classify_document_keyword
        result = classify_document_keyword("", "unknown.pdf")
        assert result.category == "Other"

    def test_identify_missing(self):
        from src.classification.document_classifier import identify_missing_categories
        from src.classification.document_classifier import ClassificationResult
        classified = [
            ClassificationResult("a.pdf", "Term Sheet", 0.9, "keyword"),
            ClassificationResult("b.pdf", "Appraisal", 0.85, "llm"),
        ]
        missing = identify_missing_categories(classified)
        assert "Loan Application" in missing
        assert "Purchase Contract" in missing
        assert "Term Sheet" not in missing
        assert "Appraisal" not in missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

