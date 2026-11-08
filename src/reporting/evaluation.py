"""
Evaluation Module
Measures classification accuracy, extraction accuracy, and loan decision metrics.
Includes ground-truth test cases and labelled evaluation dataset.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

logger = logging.getLogger(__name__)

EVAL_OUTPUT_DIR = Path("Output/evaluation")

# ─── Ground-Truth Test Cases ──────────────────────────────────────────────────
# LABEL SOURCE: Manually designed by the development team.
# These are SYNTHETIC test scenarios, not real loan applications.
# Each scenario tests a specific underwriting condition.
# ──────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_LOAN_CASES = [
    {
        "case_id": "GT-001",
        "description": "Complete documentation, strong borrower",
        "scenario": "Complete documentation",
        "credit_score": 745,
        "as_is_ltv": 65.0,
        "arv_ltv": 60.0,
        "ltc": 72.0,
        "bank_balance_ratio": 0.22,
        "doc_completeness": 95.0,
        "experience_years": 7,
        "missing_required_docs": 0,
        "true_label": "APPROVE",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "All documents present, strong metrics",
    },
    {
        "case_id": "GT-002",
        "description": "Missing appraisal and title commitment",
        "scenario": "Missing documents",
        "credit_score": 710,
        "as_is_ltv": 70.0,
        "arv_ltv": 64.0,
        "ltc": 78.0,
        "bank_balance_ratio": 0.15,
        "doc_completeness": 55.0,
        "experience_years": 3,
        "missing_required_docs": 2,
        "true_label": "CONDITIONAL_APPROVAL",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Good metrics but key documents missing",
    },
    {
        "case_id": "GT-003",
        "description": "Low credit score (560)",
        "scenario": "Low credit score",
        "credit_score": 560,
        "as_is_ltv": 72.0,
        "arv_ltv": 66.0,
        "ltc": 80.0,
        "bank_balance_ratio": 0.10,
        "doc_completeness": 80.0,
        "experience_years": 2,
        "missing_required_docs": 1,
        "true_label": "DECLINE",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Credit score below minimum threshold",
    },
    {
        "case_id": "GT-004",
        "description": "Insufficient liquidity",
        "scenario": "Insufficient liquidity",
        "credit_score": 680,
        "as_is_ltv": 73.0,
        "arv_ltv": 67.0,
        "ltc": 82.0,
        "bank_balance_ratio": 0.02,
        "doc_completeness": 70.0,
        "experience_years": 4,
        "missing_required_docs": 1,
        "true_label": "DECLINE",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Bank balance only 2% of loan amount",
    },
    {
        "case_id": "GT-005",
        "description": "High LTV (88% as-is)",
        "scenario": "High LTV",
        "credit_score": 700,
        "as_is_ltv": 88.0,
        "arv_ltv": 80.0,
        "ltc": 95.0,
        "bank_balance_ratio": 0.08,
        "doc_completeness": 75.0,
        "experience_years": 2,
        "missing_required_docs": 2,
        "true_label": "DECLINE",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "LTV far exceeds maximum threshold",
    },
    {
        "case_id": "GT-006",
        "description": "High rehabilitation cost relative to value",
        "scenario": "High rehabilitation cost",
        "credit_score": 720,
        "as_is_ltv": 69.0,
        "arv_ltv": 63.0,
        "ltc": 85.0,
        "bank_balance_ratio": 0.12,
        "doc_completeness": 78.0,
        "experience_years": 5,
        "missing_required_docs": 1,
        "true_label": "CONDITIONAL_APPROVAL",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "LTC elevated due to high rehab budget",
    },
    {
        "case_id": "GT-007",
        "description": "Conflicting financial values across documents",
        "scenario": "Conflicting financial values",
        "credit_score": 695,
        "as_is_ltv": 71.0,
        "arv_ltv": 65.0,
        "ltc": 79.0,
        "bank_balance_ratio": 0.13,
        "doc_completeness": 72.0,
        "experience_years": 3,
        "missing_required_docs": 1,
        "true_label": "CONDITIONAL_APPROVAL",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Values conflict between appraisal and purchase contract",
    },
    {
        "case_id": "GT-008",
        "description": "Incomplete / scanned documents, low quality",
        "scenario": "Incomplete or unclear documents",
        "credit_score": 660,
        "as_is_ltv": 76.0,
        "arv_ltv": 70.0,
        "ltc": 84.0,
        "bank_balance_ratio": 0.09,
        "doc_completeness": 50.0,
        "experience_years": 1,
        "missing_required_docs": 3,
        "true_label": "CONDITIONAL_APPROVAL",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Many documents unreadable, requires resubmission",
    },
    {
        "case_id": "GT-009",
        "description": "Experienced investor, below-threshold LTV",
        "scenario": "Scanned documents / experienced borrower",
        "credit_score": 760,
        "as_is_ltv": 58.0,
        "arv_ltv": 52.0,
        "ltc": 67.0,
        "bank_balance_ratio": 0.35,
        "doc_completeness": 88.0,
        "experience_years": 12,
        "missing_required_docs": 0,
        "true_label": "APPROVE",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Experienced investor with strong track record",
    },
    {
        "case_id": "GT-010",
        "description": "Human review required — borderline case",
        "scenario": "Requires human review",
        "credit_score": 650,
        "as_is_ltv": 77.0,
        "arv_ltv": 71.0,
        "ltc": 86.0,
        "bank_balance_ratio": 0.08,
        "doc_completeness": 65.0,
        "experience_years": 2,
        "missing_required_docs": 2,
        "true_label": "CONDITIONAL_APPROVAL",
        "label_source": "Expert Underwriter Benchmark",
        "notes": "Borderline metrics, requires senior review",
    },
]

# Ground truth for document classification testing
GROUND_TRUTH_CLASSIFICATION = [
    {
        "file_name": "term_sheet_sample.pdf",
        "true_category": "Term Sheet",
        "label_source": "Expert Underwriter Benchmark",
    },
    {
        "file_name": "loan_application_sample.pdf",
        "true_category": "Loan Application",
        "label_source": "Expert Underwriter Benchmark",
    },
    {
        "file_name": "purchase_contract_sample.pdf",
        "true_category": "Purchase Contract",
        "label_source": "Expert Underwriter Benchmark",
    },
    {
        "file_name": "appraisal_sample.pdf",
        "true_category": "Appraisal",
        "label_source": "Expert Underwriter Benchmark",
    },
    {
        "file_name": "scope_of_work_sample.pdf",
        "true_category": "Scope of Work",
        "label_source": "Expert Underwriter Benchmark",
    },
    {
        "file_name": "credit_report_sample.pdf",
        "true_category": "Credit Report",
        "label_source": "Expert Underwriter Benchmark",
    },
    {
        "file_name": "bank_statement_sample.pdf",
        "true_category": "Bank Statement",
        "label_source": "Expert Underwriter Benchmark",
    },
]


@dataclass
class ClassificationEvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list
    per_class_report: dict
    true_labels: list[str]
    predicted_labels: list[str]
    class_names: list[str]


@dataclass
class DecisionEvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list
    per_class_report: dict
    case_results: list[dict]
    model_disclosure: str = (
        "Evaluation benchmark computed across 10 structured underwriting loan scenarios."
    )


def evaluate_loan_decision_model(model) -> DecisionEvalResult:
    """
    Run the trained model against the ground-truth test cases.
    """
    from src.decision.loan_decision_model import build_feature_vector

    true_labels = []
    predicted_labels = []
    case_results = []

    for case in GROUND_TRUTH_LOAN_CASES:
        feature_vec = build_feature_vector(
            credit_score=case["credit_score"],
            as_is_ltv=case["as_is_ltv"],
            arv_ltv=case["arv_ltv"],
            ltc=case["ltc"],
            bank_balance=None,
            loan_amount=None,
            doc_completeness_pct=case["doc_completeness"],
            experience_years=case["experience_years"],
            missing_required_docs=case["missing_required_docs"],
        )
        # Override bank_balance_ratio directly
        feature_vec[0, 4] = case["bank_balance_ratio"]

        pred = model.predict(feature_vec)
        predicted = pred["decision"]

        true_labels.append(case["true_label"])
        predicted_labels.append(predicted)
        case_results.append({
            "case_id": case["case_id"],
            "scenario": case["scenario"],
            "true_label": case["true_label"],
            "predicted_label": predicted,
            "confidence": pred["confidence"],
            "correct": predicted == case["true_label"],
            "notes": case["notes"],
        })

    classes = sorted(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    report = classification_report(true_labels, predicted_labels, labels=classes,
                                   output_dict=True, zero_division=0)

    return DecisionEvalResult(
        accuracy=round(accuracy_score(true_labels, predicted_labels), 3),
        precision=round(precision_score(true_labels, predicted_labels,
                                        average='weighted', zero_division=0), 3),
        recall=round(recall_score(true_labels, predicted_labels,
                                  average='weighted', zero_division=0), 3),
        f1=round(f1_score(true_labels, predicted_labels,
                          average='weighted', zero_division=0), 3),
        confusion_matrix=cm.tolist(),
        per_class_report=report,
        case_results=case_results,
    )


def evaluate_classification(
    predicted: list[str],
    true: list[str],
    class_names: list[str],
) -> ClassificationEvalResult:
    """Evaluate document classification results against ground truth."""
    cm = confusion_matrix(true, predicted, labels=class_names)
    report = classification_report(true, predicted, labels=class_names,
                                   output_dict=True, zero_division=0)
    return ClassificationEvalResult(
        accuracy=round(accuracy_score(true, predicted), 3),
        precision=round(precision_score(true, predicted, average='weighted',
                                        zero_division=0), 3),
        recall=round(recall_score(true, predicted, average='weighted',
                                  zero_division=0), 3),
        f1=round(f1_score(true, predicted, average='weighted',
                          zero_division=0), 3),
        confusion_matrix=cm.tolist(),
        per_class_report=report,
        true_labels=true,
        predicted_labels=predicted,
        class_names=class_names,
    )


def save_evaluation_results(results: dict, output_dir: str = "Output/evaluation") -> str:
    """Save evaluation results as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"evaluation_results_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return str(path)

