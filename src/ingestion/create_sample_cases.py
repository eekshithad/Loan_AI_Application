"""Create 4 clean sample loan cases."""
import json
from pathlib import Path

sample_cases_dir = Path("D:/WORK/resumes/FinacPlus/Application/Input/sample_loan_cases")
sample_cases_dir.mkdir(parents=True, exist_ok=True)

for existing in sample_cases_dir.glob("*.json"):
    existing.unlink()

cases = [
    {
        "case_id": "CASE-01",
        "title": "Prime Borrower — Clean Approval",
        "borrower_entity": "Meridian Capital Investments LLC",
        "guarantor": "James A. Whitfield",
        "credit_score": 745,
        "requested_loan": 285000,
        "purchase_price": 220000,
        "as_is_value": 235000,
        "arv": 375000,
        "rehab_budget": 95000,
        "bank_balance": 72500,
        "experience_flips": 8,
        "documents_submitted": ["term_sheet", "loan_application", "purchase_contract", "appraisal"],
        "expected_decision": "APPROVE",
        "rationale": "Strong credit score, adequate equity, documented prior investor experience, clean title and contract."
    },
    {
        "case_id": "CASE-02",
        "title": "Elevated Rehab Scope — Conditional Approval",
        "borrower_entity": "Oakwood Holdings LLC",
        "guarantor": "Robert M. Vance",
        "credit_score": 685,
        "requested_loan": 320000,
        "purchase_price": 250000,
        "as_is_value": 260000,
        "arv": 450000,
        "rehab_budget": 140000,
        "bank_balance": 45000,
        "experience_flips": 3,
        "documents_submitted": ["term_sheet", "loan_application", "purchase_contract", "appraisal"],
        "expected_decision": "CONDITIONAL_APPROVAL",
        "rationale": "Satisfactory borrower profile; conditional on general contractor milestone escrow verification."
    },
    {
        "case_id": "CASE-03",
        "title": "Excessive Leverage — Declined",
        "borrower_entity": "Apex Renovations Group",
        "guarantor": "Marcus K. Bell",
        "credit_score": 620,
        "requested_loan": 420000,
        "purchase_price": 300000,
        "as_is_value": 310000,
        "arv": 440000,
        "rehab_budget": 120000,
        "bank_balance": 15000,
        "experience_flips": 1,
        "documents_submitted": ["term_sheet", "loan_application", "purchase_contract", "appraisal"],
        "expected_decision": "DECLINE",
        "rationale": "Excessive As-Is LTV (135%) and ARV LTV (95%) exceeding credit policy limits with insufficient reserves."
    },
    {
        "case_id": "CASE-04",
        "title": "Subprime Credit Profile — Declined",
        "borrower_entity": "Silverline Properties LLC",
        "guarantor": "David S. Miller",
        "credit_score": 560,
        "requested_loan": 190000,
        "purchase_price": 160000,
        "as_is_value": 175000,
        "arv": 260000,
        "rehab_budget": 60000,
        "bank_balance": 22000,
        "experience_flips": 0,
        "documents_submitted": ["term_sheet", "loan_application", "purchase_contract", "appraisal"],
        "expected_decision": "DECLINE",
        "rationale": "Credit score (560) is below minimum threshold (580) and guarantor lacks prior investment experience."
    }
]

for c in cases:
    fname = f"{c['case_id'].lower()}_{c['expected_decision'].lower()}.json"
    with open(sample_cases_dir / fname, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)

print("Created 4 sample loan cases in Input/sample_loan_cases")
