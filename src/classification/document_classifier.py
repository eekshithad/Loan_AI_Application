"""
Document Classification Module
Uses keyword heuristics + LLM (llama3.1 via Ollama) to classify
each PDF into one of the 16 loan document categories.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DOCUMENT_CATEGORIES = [
    "Term Sheet",
    "Loan Application",
    "Purchase Contract",
    "Appraisal",
    "Scope of Work",
    "Title Commitment",
    "Insurance Binder",
    "Flood Certificate",
    "Credit Report",
    "Bank Statement",
    "Operating Agreement",
    "Certificate of Good Standing",
    "Government ID",
    "Borrower Track Record",
    "Background Check",
    "Other",
]

# Keyword hints for each category (used as fast pre-filter)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Term Sheet": [
        "term sheet", "loan terms", "interest rate", "origination fee",
        "maturity date", "bridge loan terms", "commitment letter",
    ],
    "Loan Application": [
        "loan application", "uniform residential", "1003", "borrower application",
        "application for mortgage", "urla",
    ],
    "Purchase Contract": [
        "purchase contract", "purchase agreement", "real estate purchase",
        "sale agreement", "buyer", "seller", "purchase price", "earnest money",
        "contract for sale",
    ],
    "Appraisal": [
        "appraisal report", "uniform residential appraisal", "as-is value",
        "after repair value", "arv", "appraised value", "market value",
        "comparable sales", "urar", "appraisal",
    ],
    "Scope of Work": [
        "scope of work", "rehabilitation", "renovation budget", "repair schedule",
        "contractor bid", "construction budget", "work items", "rehab cost",
    ],
    "Title Commitment": [
        "title commitment", "title insurance", "schedule a", "schedule b",
        "commitment for title", "title search", "lien search",
    ],
    "Insurance Binder": [
        "insurance binder", "certificate of insurance", "homeowner's insurance",
        "property insurance", "policy number", "named insured", "acord",
        "declarations page", "insurance certificate",
    ],
    "Flood Certificate": [
        "flood certificate", "national flood insurance", "nfip",
        "flood zone", "flood determination", "fema", "flood elevation",
    ],
    "Credit Report": [
        "credit report", "fico score", "credit score", "equifax", "experian",
        "transunion", "credit history", "payment history", "credit bureau",
    ],
    "Bank Statement": [
        "bank statement", "account statement", "checking account", "savings account",
        "account balance", "deposits", "withdrawals", "account number",
    ],
    "Operating Agreement": [
        "operating agreement", "limited liability company", "llc agreement",
        "member agreement", "articles of organization", "membership interest",
    ],
    "Certificate of Good Standing": [
        "certificate of good standing", "certificate of existence",
        "good standing", "secretary of state", "corporate status",
    ],
    "Government ID": [
        "driver's license", "passport", "state id", "government issued id",
        "identification", "social security",
    ],
    "Borrower Track Record": [
        "track record", "investment history", "real estate portfolio",
        "completed projects", "flips completed", "prior projects",
    ],
    "Background Check": [
        "background check", "background report", "criminal history",
        "public records", "sex offender", "criminal background",
    ],
}


@dataclass
class ClassificationResult:
    file_name: str
    category: str
    confidence: float          # 0.0 – 1.0
    method: str                # "keyword", "llm", "keyword+llm"
    keyword_matches: list[str] = field(default_factory=list)
    page_evidence: dict[int, str] = field(default_factory=dict)  # page → snippet
    llm_reasoning: str = ""
    error: Optional[str] = None


def _keyword_score(text: str) -> dict[str, float]:
    """Return a score per category based on keyword hits in the text."""
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = [kw for kw in kws if kw in text_lower]
        scores[cat] = len(hits) / max(len(kws), 1)
    return scores


def _top_keyword_category(scores: dict[str, float]) -> tuple[str, float, list[str]]:
    """Return (best_category, score, matched_keywords)."""
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    if best_score == 0:
        return "Other", 0.0, []
    matched = [
        kw for kw in CATEGORY_KEYWORDS.get(best_cat, [])
        if kw in "placeholder"  # filled properly below
    ]
    return best_cat, best_score, []


def classify_document_keyword(text: str, file_name: str = "") -> ClassificationResult:
    """Fast keyword-based classification."""
    scores = _keyword_score(text)
    text_lower = text.lower()

    # Also check filename with normalized spacing
    fname_lower = file_name.lower().replace("_", " ").replace("-", " ")
    for cat, kws in CATEGORY_KEYWORDS.items():
        if cat.lower() in fname_lower or any(kw in fname_lower for kw in kws):
            scores[cat] = scores.get(cat, 0) + 0.6

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    if best_score < 0.05:
        best_cat = "Other"
        best_score = 0.1

    matched_kws = [kw for kw in CATEGORY_KEYWORDS.get(best_cat, [])
                   if kw in text_lower]

    # Normalize confidence (keyword alone maxes at 0.75)
    confidence = min(best_score * 1.5, 0.75)

    return ClassificationResult(
        file_name=file_name,
        category=best_cat,
        confidence=round(confidence, 2),
        method="keyword",
        keyword_matches=matched_kws,
    )


def classify_document_llm(text: str, file_name: str, ollama_client) -> ClassificationResult:
    """
    LLM-based classification using llama3.1.
    Returns ClassificationResult. Falls back to keyword if LLM fails.
    """
    # Truncate to first 3000 chars to keep prompt manageable
    truncated = text[:3000].strip()
    if not truncated:
        truncated = "(No readable text found in document)"

    categories_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(DOCUMENT_CATEGORIES))

    prompt = f"""You are a loan underwriting document classifier.

The document filename is: "{file_name}"

Here are the first part of the document's text content:
---
{truncated}
---

Classify this document into exactly ONE of these categories:
{categories_list}

Respond in this exact format (nothing else):
CATEGORY: <category name exactly as listed>
CONFIDENCE: <high|medium|low>
REASON: <one sentence explaining why>
"""

    try:
        response = ollama_client.chat(prompt)
        lines = response.strip().split("\n")

        category = "Other"
        confidence_str = "low"
        reason = ""

        for line in lines:
            clean_line = line.strip().lstrip("*# ")
            if clean_line.upper().startswith("CATEGORY:"):
                raw_cat = clean_line[len("CATEGORY:"):].strip().strip("*\"' ")
                # Try exact match first
                for known in DOCUMENT_CATEGORIES:
                    if known.lower() == raw_cat.lower():
                        category = known
                        break
                else:
                    # Try flexible substring match (e.g. "Appraisal Report" -> "Appraisal")
                    for known in DOCUMENT_CATEGORIES:
                        if known.lower() in raw_cat.lower() or raw_cat.lower() in known.lower():
                            category = known
                            break
                    else:
                        category = "Other"
            elif clean_line.upper().startswith("CONFIDENCE:"):
                confidence_str = clean_line[len("CONFIDENCE:"):].strip().lower().strip("*\"' ")
            elif clean_line.upper().startswith("REASON:"):
                reason = clean_line[len("REASON:"):].strip().strip("*\"' ")

        conf_map = {"high": 0.90, "medium": 0.70, "low": 0.50}
        confidence = conf_map.get(confidence_str, 0.50)

        return ClassificationResult(
            file_name=file_name,
            category=category,
            confidence=confidence,
            method="llm",
            llm_reasoning=reason,
        )

    except Exception as exc:
        logger.warning(f"LLM classification failed for '{file_name}': {exc}")
        # Fall back to keyword
        result = classify_document_keyword(text, file_name)
        result.error = f"LLM failed: {exc}, used keyword fallback"
        return result


def classify_document(
    text: str,
    file_name: str,
    ollama_client=None,
    use_llm: bool = True
) -> ClassificationResult:
    """
    Classify a document using keyword heuristics first,
    then optionally confirm/override with LLM.
    """
    keyword_result = classify_document_keyword(text, file_name)

    if use_llm and ollama_client is not None:
        try:
            llm_result = classify_document_llm(text, file_name, ollama_client)
            # If LLM classified successfully, use it
            if llm_result.category != "Other":
                if llm_result.category == keyword_result.category:
                    combined_conf = min((llm_result.confidence + keyword_result.confidence) / 2 + 0.1, 1.0)
                    llm_result.confidence = round(combined_conf, 2)
                    llm_result.method = "keyword+llm"
                    llm_result.keyword_matches = keyword_result.keyword_matches
                return llm_result
            elif keyword_result.category != "Other":
                # Fall back to keyword if LLM returned Other
                keyword_result.llm_reasoning = "Keyword matched category"
                return keyword_result
            return llm_result
        except Exception as exc:
            logger.warning(f"LLM classification error: {exc}")

    return keyword_result


def identify_missing_categories(
    classified_docs: list[ClassificationResult],
    required_categories: list[str] | None = None,
) -> list[str]:
    """Return list of expected but missing document categories."""
    if required_categories is None:
        required_categories = [
            "Term Sheet",
            "Loan Application",
            "Purchase Contract",
            "Appraisal",
        ]
    found = {r.category for r in classified_docs}
    return [cat for cat in required_categories if cat not in found]

