# Senior AI Engineer Underwriting System — Project Write-Up


## 1. Executive Summary & Approach

Commercial bridge lending ("fix-and-flip") requires rapid, objective credit risk evaluation across complex, heterogeneous loan packages. In this project, I designed and built an **AI-Driven Loan Underwriting Application** from the ground up to assist human credit underwriters by automating document classification, structured data extraction, deterministic financial metrics calculation, and machine-learning-based loan recommendation.

### System Architecture
The application is architected around a clear separation of concerns:
```
Loan Documents (PDF)
        │
        ▼
[1] Ingestion Layer (PyMuPDF + Tesseract OCR)
        │
        ▼
[2] Classification Engine (Keyword Heuristics + local llama3.1)
        │
        ▼
[3] Information Extraction (llama3.1 structured JSON extraction)
        │
        ├──► [4] Financial Calculation Engine (Deterministic Python)
        │         • As-Is LTV, ARV LTV, LTC, Borrower Equity
        │
        ├──► [5] Document Validation Rules (Deterministic Python)
        │         • Missing docs, policy threshold violations, value conflicts
        │
        └──► [6] Decision Engine (Scikit-Learn Random Forest Classifier)
                  • APPROVE / CONDITIONAL APPROVAL / DECLINE
                       │
                       ▼
             [7] Underwriter Explanation (llama3.1 synthesis)
                       │
                       ▼
             [8] Underwriting Report (ReportLab PDF + JSON Export)
```

**Key Architectural Principle:** High-stakes financial arithmetic (LTV, LTC, equity) is executed strictly in deterministic Python code—never by an LLM. The LLM (`llama3.1` via local Ollama) is leveraged exclusively where it excels: semantic document classification, unstructured entity extraction, and synthesizing plain-English explanations.

---

## 2. Answers to Core Underwriting Questions

### Question 1: Is the property worth enough?
- **Analysis:** The system extracts the **As-Is Market Value** (\$235,000) and **After-Repair Value (ARV)** (\$375,000) from the appraisal report and calculates both **As-Is LTV** (121.3%) and **ARV LTV** (76.0%).
- **Underwriting Finding:** While the initial As-Is LTV is high because this is a renovation project, the ARV LTV of 76.0% demonstrates collateral adequacy upon completion of improvements. The property is located in FEMA Flood Zone X (no flood hazard), and Peachtree Title Services confirmed clear fee-simple title with no tax liens.

### Question 2: Can the borrower pull this off?
- **Analysis:** The borrower has a qualifying middle FICO score of **718** (Tier 1 prime), **$72,500** in verified liquid checking reserves, and a documented **track record of 8 completed fix-and-flip projects** over 5 years generating \$308,000 in net profit.
- **Underwriting Finding:** The borrower possesses adequate operational experience and capital reserves to service monthly interest and manage the planned 90-day renovation.

### Question 3: Is the paperwork clean?
- **Analysis:** The pipeline cross-references figures across 12 distinct document categories. 
- **Underwriting Finding:** The \$220,000 purchase price matches between the loan application and purchase contract; the \$95,000 rehab budget aligns between the scope of work and appraisal; hazard/builder's risk insurance is bound through Southern Shield Insurance; and the corporate operating agreement confirms managerial borrowing authority.

---

## 3. Answers to Technical Architecture Questions

### Question 1: Which documents and which fields actually drive the decision? How did you find out?
- **Analysis:** Feature importance analysis of our Random Forest model and empirical credit policy guidelines reveal that the decision is driven by **4 core documents**:
  1. *Appraisal:* As-Is Market Value and After-Repair Value (ARV).
  2. *Loan Application:* Borrower entity, Guarantor FICO score, liquid cash reserves, and completed project history.
  3. *Purchase Contract:* Contract purchase price, earnest money, and closing timeframe.
  4. *Term Sheet:* Requested loan amount, interest rate, and holdback terms.
- **Finding:** Quantitative features (ARV LTV, LTC, Credit Score, Liquidity Ratio) account for over 80% of decision variance. Ancillary documents (Title, Insurance, Flood) function as binary eligibility gates (conditions precedent) rather than continuous risk weights.

### Question 2: Which facts should be computed from extracted values rather than taken directly from a document?
- **Analysis:** Key leverage and risk metrics must **always be computed deterministically in code**:
  - `As-Is LTV = (Loan Amount / As-Is Value) * 100`
  - `ARV LTV = (Loan Amount / ARV) * 100`
  - `LTC = (Loan Amount / [Purchase Price + Rehab Budget]) * 100`
  - `Borrower Equity = Total Project Cost - Loan Amount`
- **Finding:** Documents often present conflicting or outdated ratios (e.g., broker pitch decks quoting optimistic ARVs). Recalculating ratios directly from independent primary sources guarantees mathematical integrity and prevents unstated cost overruns.

### Question 3: Where is an LLM the right tool, and where is a classical ML model, an OCR engine, or a simple parser better?
- **OCR Engine (PyMuPDF / Tesseract):** Superior for low-level raster image parsing and coordinate-aware text extraction from scanned forms.
- **Pattern / Regex Parser:** Ideal for standardized forms (e.g., Form 1003, ACORD insurance binders) where layout is fixed; executes in < 10ms with zero hallucination risk.
- **Local LLM (`llama3.1` via Ollama):** Best for unstructured semantic reasoning: mapping heterogeneous document types, parsing varied contractor bids, and synthesizing multi-factor underwriter justifications.
- **Classical ML (Random Forest):** Superior for credit risk scoring. Provides reproducible, calibrated class probabilities and auditable feature weights required for regulatory compliance and credit committee governance.

### Question 4: How do you stop the system inventing values that are not in the documents?
- **Defense-in-Depth Guardrails:**
  1. *Explicit "Not Found" Default:* Missing entities are assigned `Not Found` / `null` rather than imputed.
  2. *Source Attribution:* Every extracted entity captures its originating `source_document` and 1-based `source_page`.
  3. *Zero-Temperature JSON Constrained Prompting:* LLM system prompts strictly enforce: *"Return ONLY valid JSON. If a value is missing, return null. Never extrapolate or guess."*

### Question 5: When should the system refuse to decide and hand the file to a human?
- **Human Escalation Triggers:**
  - Prediction confidence falls below 60%.
  - Any core required document is missing.
  - High-severity cross-document discrepancies (e.g., Contract Price != Application Price by > 5%).
  - Borderline credit scores (620–660) coupled with high leverage (LTC > 85%).
  - Environmental red flags (e.g., property located in Special Flood Hazard Area Zone A/V without flood insurance binder).

### Question 6: How would you explain a decline to an underwriter so they can trust or challenge it?
- **Auditable Adverse Action Presentation:**
  - Clearly state the primary policy breaches (e.g., *"LTC of 94.2% exceeds maximum ceiling of 90.0%"* or *"Liquid reserves of $12,000 represent only 3.8% of requested facility, violating the 10% liquidity policy"*).
  - Provide direct hyperlinks/citations to the exact document and page number where the offending data originated.
  - Display top model factor weights so underwriters can identify mitigating adjustments (e.g., injecting additional cash equity or adding an experienced co-guarantor).

---

## 4. Trade-Offs Made

1. **Local LLM (`llama3.1` via Ollama) vs. Hosted APIs (Claude/OpenAI):**
   - *Trade-Off:* Local execution is slower on consumer hardware (~12–18s per document on CPU) compared to hosted cloud APIs.
   - *Rationale:* Selected local Ollama to ensure complete data sovereignty, eliminate per-token cloud costs, and guarantee zero PII transmission outside the local runtime.
2. **Deterministic Financial Math vs. LLM-Based Calculations:**
   - *Trade-Off:* Requires explicit Python formulas and rigid regex/schema parsing rather than free-form prompting.
   - *Rationale:* Financial lending decisions require absolute mathematical reproducibility; LLMs cannot be trusted with floating-point loan arithmetic.
3. **Random Forest vs. Deep Learning for Credit Decision:**
   - *Trade-Off:* Tabular Random Forest cannot directly ingest raw unstructured text.
   - *Rationale:* Loan decisions require explainability. Random Forest provides exact feature importance weights and calibrated class probabilities required for human credit committee review.

---

## 5. What I Would Do With Another Two Weeks

1. **Automated Cross-Document Discrepancy Matrix:** Expand the conflict detector into an interactive matrix highlighting exact line-item discrepancies between contractor bids and appraiser cost estimates.
2. **Multi-Column OCR / Table Extraction with LayoutLM:** Integrate a specialized document layout model for dense, multi-column bank ledgers and title schedules.
3. **Automated Adverse Action Notice Generation:** Add compliance-grade adverse action letter generation per Equal Credit Opportunity Act (ECOA) standards when a loan is declined.
4. **Batch Portfolio Processing & Parallel Workers:** Implement Celery/Redis background worker queues to process 100+ loan packages simultaneously across multi-GPU nodes.

---

## 6. AI Usage Notes

In building this application, I utilized **Google Antigravity** as my primary agentic AI coding assistant.
- **How I Used Antigravity:**
  - I framed the business requirements, designed the modular architecture, defined the financial validation formulas, and formulated the prompt engineering specifications.
  - I directed Antigravity to generate boilerplate modules, configure the virtual environment, install package dependencies, construct the ReportLab PDF rendering pipeline, and wire up the Streamlit UI components.
- **Where I Intervened, Modified, or Rejected AI Output:**
  - *NumPy 2.x Compatibility:* Python 3.13 lacked pre-built binary wheels for older NumPy versions. I corrected the dependency constraints to use compatible modern wheels (`numpy>=2.1.0`).
  - *Report Formatting & Prompt Engineering:* I revised the prompt templates to enforce strict JSON schemas and removed unnecessary disclaimer artifacts to produce clean, authentic enterprise-grade reports.
  - *Model Separation:* Ensured the Random Forest decision module operated independently from LLM explanations to preserve auditability.

---

## 7. Model Evaluation & Benchmark Results

The decision engine was evaluated against 10 diverse institutional benchmark scenarios (`GT-001` through `GT-010`) covering complete documentation, high leverage, low liquidity, subprime credit, and scanned/unclear files:

| Evaluation Metric | Benchmark Performance | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **90.0%** | 9 out of 10 test scenarios correctly classified |
| **Precision (Weighted)** | **92.5%** | High positive predictive value across all three credit decisions |
| **Recall (Weighted)** | **90.0%** | Comprehensive sensitivity across approvals, conditions, and declines |
| **F1 Score (Weighted)** | **90.2%** | Balanced precision-recall trade-off |
| **Processing Speed (CPU)** | **~1.5s / file** | CPU-optimized heuristic pipeline; ~18s / file with local `llama3.1` |

### Error Analysis & Model Divergence:
- **Case GT-008 (*Incomplete / Scanned Documents*):** The Random Forest model predicted `DECLINE` due to 3 missing required documents and 50% document completeness, whereas the benchmark credit policy permitted `CONDITIONAL_APPROVAL` subject to document resubmission. This demonstrates the model's cautious risk bias in the presence of severe data gaps.

---

## 8. Data Note & Dataset Provenance

- **Loan Document Package:** 12 complete, multi-page PDF loan documents were prepared based on standard commercial lending templates:
  1. `term_sheet_sample.pdf` (Commercial Bridge Loan Term Sheet)
  2. `loan_application_sample.pdf` (Uniform Commercial Application / Form 1003)
  3. `purchase_contract_sample.pdf` (Georgia Association of REALTORS Standard Agreement)
  4. `appraisal_sample.pdf` (Uniform Residential Appraisal Report Form 1004)
  5. `scope_of_work_sample.pdf` (Itemized Contractor Trade Schedule)
  6. `credit_report_sample.pdf` (Tri-Merge Equifax/Experian/TransUnion Report)
  7. `bank_statement_sample.pdf` (Commercial Depository Checking Statement)
  8. `title_commitment_sample.pdf` (ALTA Plain Language Title Commitment)
  9. `insurance_binder_sample.pdf` (ACORD 27 Evidence of Property Insurance)
  10. `flood_certificate_sample.pdf` (FEMA Standard Flood Hazard Determination Form 086-0-32)
  11. `operating_agreement_sample.pdf` (Articles of Organization & LLC Operating Agreement)
  12. `borrower_track_record_sample.pdf` (Historical SFR Renovation Disposition Schedule)
- **Evaluation Dataset:** 10 diverse, structured loan underwriting cases (`GT-001` to `GT-010`) covering prime approval, missing appraisal/title, severe credit impairment, insufficient liquidity, and high LTV/LTC. All ground-truth labels were established according to standard commercial credit committee criteria.
