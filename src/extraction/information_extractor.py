"""
Information Extraction Module
Extracts borrower, loan, and property information from classified documents
using LLM (llama3.1) with page-level source attribution.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedField:
    """A single extracted value with its source."""
    value: Any
    source_document: str
    source_page: int
    confidence: str  # "high", "medium", "low", "not_found"
    raw_text: str = ""


NOT_FOUND = "Not Found"


def _make_not_found(field_name: str) -> ExtractedField:
    return ExtractedField(
        value=NOT_FOUND,
        source_document="N/A",
        source_page=0,
        confidence="not_found",
    )


@dataclass
class BorrowerInfo:
    name: ExtractedField = field(default_factory=lambda: _make_not_found("name"))
    entity_type: ExtractedField = field(default_factory=lambda: _make_not_found("entity_type"))
    credit_score: ExtractedField = field(default_factory=lambda: _make_not_found("credit_score"))
    employment: ExtractedField = field(default_factory=lambda: _make_not_found("employment"))
    bank_balance: ExtractedField = field(default_factory=lambda: _make_not_found("bank_balance"))
    experience_years: ExtractedField = field(default_factory=lambda: _make_not_found("experience_years"))
    projects_completed: ExtractedField = field(default_factory=lambda: _make_not_found("projects_completed"))


@dataclass
class LoanInfo:
    loan_amount: ExtractedField = field(default_factory=lambda: _make_not_found("loan_amount"))
    purchase_price: ExtractedField = field(default_factory=lambda: _make_not_found("purchase_price"))
    as_is_value: ExtractedField = field(default_factory=lambda: _make_not_found("as_is_value"))
    arv: ExtractedField = field(default_factory=lambda: _make_not_found("arv"))
    rehab_budget: ExtractedField = field(default_factory=lambda: _make_not_found("rehab_budget"))
    property_address: ExtractedField = field(default_factory=lambda: _make_not_found("property_address"))
    interest_rate: ExtractedField = field(default_factory=lambda: _make_not_found("interest_rate"))
    loan_term_months: ExtractedField = field(default_factory=lambda: _make_not_found("loan_term_months"))
    origination_fee: ExtractedField = field(default_factory=lambda: _make_not_found("origination_fee"))


@dataclass
class SupportingInfo:
    insurance_carrier: ExtractedField = field(default_factory=lambda: _make_not_found("insurance_carrier"))
    insurance_amount: ExtractedField = field(default_factory=lambda: _make_not_found("insurance_amount"))
    flood_zone: ExtractedField = field(default_factory=lambda: _make_not_found("flood_zone"))
    title_issues: ExtractedField = field(default_factory=lambda: _make_not_found("title_issues"))
    property_type: ExtractedField = field(default_factory=lambda: _make_not_found("property_type"))
    bedrooms: ExtractedField = field(default_factory=lambda: _make_not_found("bedrooms"))
    square_footage: ExtractedField = field(default_factory=lambda: _make_not_found("square_footage"))


@dataclass
class ExtractionResult:
    borrower: BorrowerInfo = field(default_factory=BorrowerInfo)
    loan: LoanInfo = field(default_factory=LoanInfo)
    supporting: SupportingInfo = field(default_factory=SupportingInfo)
    raw_extractions: list[dict] = field(default_factory=list)


EXTRACTION_PROMPTS = {
    "Term Sheet": """Extract from this term sheet document the following fields.
Return ONLY a JSON object with these keys:
{
  "loan_amount": <number or null>,
  "interest_rate": <string or null>,
  "loan_term_months": <number or null>,
  "origination_fee": <string or null>,
  "property_address": <string or null>,
  "arv": <number or null>,
  "as_is_value": <number or null>
}
If a value is not found, use null.
Document text:
""",
    "Loan Application": """Extract from this loan application document the following fields.
Return ONLY a JSON object with these keys:
{
  "borrower_name": <string or null>,
  "entity_type": <string or null>,
  "loan_amount": <number or null>,
  "purchase_price": <number or null>,
  "property_address": <string or null>,
  "employment": <string or null>,
  "rehab_budget": <number or null>,
  "arv": <number or null>,
  "bank_balance": <number or null>,
  "experience_years": <number or null>,
  "projects_completed": <number or null>
}
If a value is not found, use null.
Document text:
""",
    "Appraisal": """Extract from this appraisal document the following fields.
Return ONLY a JSON object with these keys:
{
  "as_is_value": <number or null>,
  "arv": <number or null>,
  "property_address": <string or null>,
  "property_type": <string or null>,
  "bedrooms": <number or null>,
  "square_footage": <number or null>
}
If a value is not found, use null.
Document text:
""",
    "Purchase Contract": """Extract from this purchase contract the following fields.
Return ONLY a JSON object with these keys:
{
  "purchase_price": <number or null>,
  "property_address": <string or null>,
  "buyer_name": <string or null>,
  "seller_name": <string or null>
}
If a value is not found, use null.
Document text:
""",
    "Scope of Work": """Extract from this scope of work / renovation budget document the following fields.
Return ONLY a JSON object with these keys:
{
  "rehab_budget": <number or null>,
  "property_address": <string or null>
}
If a value is not found, use null.
Document text:
""",
    "Credit Report": """Extract from this credit report the following fields.
Return ONLY a JSON object with these keys:
{
  "borrower_name": <string or null>,
  "credit_score": <number or null>,
  "credit_bureau": <string or null>
}
If a value is not found, use null.
Document text:
""",
    "Bank Statement": """Extract from this bank statement the following fields.
Return ONLY a JSON object with these keys:
{
  "account_holder": <string or null>,
  "bank_balance": <number or null>,
  "bank_name": <string or null>
}
If a value is not found, use null.
Document text:
""",
    "Insurance Binder": """Extract from this insurance binder or certificate of insurance the following fields.
Return ONLY a JSON object with these keys:
{
  "insurance_carrier": <string or null>,
  "insurance_amount": <number or null>,
  "named_insured": <string or null>,
  "policy_number": <string or null>
}
If a value is not found, use null.
Document text:
""",
    "Flood Certificate": """Extract from this flood certificate/determination the following fields.
Return ONLY a JSON object with these keys:
{
  "flood_zone": <string or null>,
  "property_address": <string or null>,
  "nfip_required": <boolean or null>
}
If a value is not found, use null.
Document text:
""",
    "Title Commitment": """Extract from this title commitment the following fields.
Return ONLY a JSON object with these keys:
{
  "property_address": <string or null>,
  "title_issues": <string or null>,
  "title_company": <string or null>
}
If a value is not found, use null.
Document text:
""",
    "Borrower Track Record": """Extract from this borrower track record the following fields.
Return ONLY a JSON object with these keys:
{
  "borrower_name": <string or null>,
  "projects_completed": <number or null>,
  "experience_years": <number or null>
}
If a value is not found, use null.
Document text:
""",
}

DEFAULT_PROMPT = """Extract all relevant loan underwriting information from this document.
Return ONLY a JSON object with any relevant fields you can find.
Document text:
"""


def _parse_money(value: Any) -> Optional[float]:
    """Parse a monetary string or number to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = re.sub(r'[$,\s]', '', value)
        try:
            return float(clean)
        except ValueError:
            return None
    return None


def _extract_via_llm(text: str, doc_category: str, ollama_client) -> dict:
    """Use LLM to extract structured data from document text."""
    prompt_template = EXTRACTION_PROMPTS.get(doc_category, DEFAULT_PROMPT)
    # Truncate to manageable size
    truncated = text[:4000].strip()
    full_prompt = f"{prompt_template}\n{truncated}"

    system = (
        "You are a precise document data extractor. "
        "Return ONLY valid JSON, no explanation, no markdown code blocks. "
        "Use null for missing values."
    )

    try:
        raw = ollama_client.chat(full_prompt, system=system)
        # Clean up LLM response - remove markdown if present
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        # Find JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except Exception as exc:
        logger.warning(f"LLM extraction failed for category '{doc_category}': {exc}")
        return {}


def _find_page_for_value(pages, search_term: str) -> int:
    """Find the page number where a value first appears."""
    if not search_term or search_term == NOT_FOUND:
        return 0
    search_lower = str(search_term).lower()[:30]
    for page in pages:
        if search_lower in page.text.lower():
            return page.page_number
    return 1  # Default to page 1 if not specifically found


def _extract_via_patterns(text: str, doc_category: str) -> dict:
    """
    Fast regex & line-based extraction fallback for key underwriting figures.
    Optimized for CPU performance and robust pattern matching without LLM latency.
    """
    data = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def match_kv(keys: list[str]) -> Optional[str]:
        for i, line in enumerate(lines):
            l_lower = line.lower()
            for k in keys:
                if k.lower() in l_lower:
                    if ":" in line:
                        after = line.split(":", 1)[1].strip()
                        if after:
                            return after
                    if i + 1 < len(lines):
                        nxt = lines[i + 1].strip()
                        if nxt and not any(c in nxt.lower() for c in [":", "section", "page", "table", "form"]):
                            return nxt
        return None

    # Borrower Name / Entity
    b_name = match_kv(["Borrower Name / Entity", "Borrower Entity", "Purchaser / Buyer", "Borrower Name"])
    if b_name:
        data["borrower_name"] = b_name

    # Loan Amount
    l_amt = match_kv(["Requested Loan Amount", "Facility Amount", "Loan Amount", "Principal Amount"])
    if l_amt:
        parsed = _parse_money(l_amt)
        if parsed:
            data["loan_amount"] = parsed

    # Purchase Price
    p_price = match_kv(["Contract Purchase Price", "Purchase Price"])
    if p_price:
        parsed = _parse_money(p_price)
        if parsed:
            data["purchase_price"] = parsed

    # As-Is Value
    as_is = match_kv(["As-Is Appraised Value", "As-Is Market Value", "As-Is Value", "Appraised Value"])
    if as_is:
        parsed = _parse_money(as_is)
        if parsed:
            data["as_is_value"] = parsed

    # ARV
    arv = match_kv(["After Repair Value (ARV)", "After-Repair Value (ARV)", "Prospective Value Upon Completion (ARV)", "ARV"])
    if arv:
        parsed = _parse_money(arv)
        if parsed:
            data["arv"] = parsed

    # Rehab Budget
    rehab = match_kv(["Planned Rehab Budget", "Rehabilitation Budget", "Rehab Budget", "Renovation Budget"])
    if rehab:
        parsed = _parse_money(rehab)
        if parsed:
            data["rehab_budget"] = parsed

    # Credit Score
    cs = match_kv(["Principal Guarantor Middle FICO", "Credit Score", "FICO Score", "Qualifying FICO", "Middle FICO", "FICO"])
    if cs:
        m = re.search(r'\b([3-8][0-9]{2})\b', cs)
        if m:
            data["credit_score"] = int(m.group(1))

    # Bank Balance
    bal = match_kv(["Current Liquid Operating Reserves", "Bank Balance", "Liquid Reserves", "Verified Checking", "Liquid Assets", "Total Liquidity"])
    if bal:
        parsed = _parse_money(bal)
        if parsed:
            data["bank_balance"] = parsed

    # Experience / Completed Flips
    exp = match_kv(["Completed Projects", "Completed Flips", "Years in Real Estate Investment", "Prior Investment Projects", "Experience Years", "Track Record"])
    if exp:
        m = re.search(r'\b([0-9]{1,2})\b', exp)
        if m:
            data["experience_years"] = int(m.group(1))
            data["projects_completed"] = int(m.group(1))

    # Property Address
    addr = match_kv(["Collateral Address", "Property Address", "Subject Property", "Collateral Property"])
    if addr:
        data["property_address"] = addr

    return data


def extract_from_document(
    pdf_doc,
    category: str,
    ollama_client,
    use_llm: bool = True,
) -> dict:
    """
    Extract information from a single classified PDF document.
    Returns raw extraction dict. Supports fast CPU mode and LLM mode.
    """
    if not pdf_doc.full_text.strip():
        return {"_category": category, "_file": pdf_doc.file_name, "_empty": True}

    pattern_data = _extract_via_patterns(pdf_doc.full_text, category)

    if use_llm and ollama_client is not None and getattr(ollama_client, 'model', None):
        try:
            llm_data = _extract_via_llm(pdf_doc.full_text, category, ollama_client)
            # Combine: pattern data provides robust foundation, llm overlays nuances
            for k, v in pattern_data.items():
                if k not in llm_data or llm_data[k] is None:
                    llm_data[k] = v
            extracted = llm_data
        except Exception:
            extracted = pattern_data
    else:
        extracted = pattern_data

    extracted["_category"] = category
    extracted["_file"] = pdf_doc.file_name
    extracted["_pages"] = pdf_doc.total_pages
    return extracted


def merge_extractions(
    extractions: list[tuple[dict, object]],  # (extracted_dict, pdf_doc)
) -> ExtractionResult:
    """
    Merge extractions from multiple documents into a unified ExtractionResult.
    Later documents override earlier ones only if the field is currently "Not Found".
    """
    result = ExtractionResult()

    for ext, pdf_doc in extractions:
        file_name = ext.get("_file", "Unknown")
        pages = getattr(pdf_doc, 'pages', [])

        # Helper to build ExtractedField
        def make_field(raw_val, unit=None) -> Optional[ExtractedField]:
            if raw_val is None:
                return None
            val = raw_val
            if unit == "money":
                parsed = _parse_money(raw_val)
                if parsed is None:
                    return None
                val = parsed
            page = _find_page_for_value(pages, str(raw_val)[:30] if raw_val else "")
            return ExtractedField(
                value=val,
                source_document=file_name,
                source_page=page,
                confidence="medium",
                raw_text=str(raw_val),
            )

        def set_if_missing(obj, attr, raw_val, unit=None):
            current = getattr(obj, attr, None)
            if current is None or (hasattr(current, 'value') and current.value == NOT_FOUND):
                f = make_field(raw_val, unit)
                if f:
                    setattr(obj, attr, f)

        # Borrower info
        set_if_missing(result.borrower, "name",
                       ext.get("borrower_name") or ext.get("account_holder") or ext.get("named_insured"))
        set_if_missing(result.borrower, "entity_type", ext.get("entity_type"))
        set_if_missing(result.borrower, "credit_score", ext.get("credit_score"))
        set_if_missing(result.borrower, "employment", ext.get("employment"))
        set_if_missing(result.borrower, "bank_balance", ext.get("bank_balance"), unit="money")
        set_if_missing(result.borrower, "experience_years", ext.get("experience_years"))
        set_if_missing(result.borrower, "projects_completed", ext.get("projects_completed"))

        # Loan info
        set_if_missing(result.loan, "loan_amount", ext.get("loan_amount"), unit="money")
        set_if_missing(result.loan, "purchase_price", ext.get("purchase_price"), unit="money")
        set_if_missing(result.loan, "as_is_value", ext.get("as_is_value"), unit="money")
        set_if_missing(result.loan, "arv", ext.get("arv"), unit="money")
        set_if_missing(result.loan, "rehab_budget", ext.get("rehab_budget"), unit="money")
        set_if_missing(result.loan, "property_address",
                       ext.get("property_address"))
        set_if_missing(result.loan, "interest_rate", ext.get("interest_rate"))
        set_if_missing(result.loan, "loan_term_months", ext.get("loan_term_months"))
        set_if_missing(result.loan, "origination_fee", ext.get("origination_fee"))

        # Supporting info
        set_if_missing(result.supporting, "insurance_carrier", ext.get("insurance_carrier"))
        set_if_missing(result.supporting, "insurance_amount",
                       ext.get("insurance_amount"), unit="money")
        set_if_missing(result.supporting, "flood_zone", ext.get("flood_zone"))
        set_if_missing(result.supporting, "title_issues", ext.get("title_issues"))
        set_if_missing(result.supporting, "property_type", ext.get("property_type"))
        set_if_missing(result.supporting, "bedrooms", ext.get("bedrooms"))
        set_if_missing(result.supporting, "square_footage", ext.get("square_footage"))

        result.raw_extractions.append(ext)

    return result

