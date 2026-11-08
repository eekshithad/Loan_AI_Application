# AI-Driven Loan Underwriting Application

An intelligent, production-ready system for automated real estate bridge loan underwriting. Ingests heterogeneous loan PDF packages, classifies document types, extracts key financial facts with exact page attribution, executes deterministic financial ratio calculations, and generates an AI/ML credit recommendation with human-verifiable explanations.

---

## Quick Start (Single Command)

### 1. Environment Activation
```powershell
# Navigate to the application root
cd D:\WORK\resumes\FinacPlus\Application

# Activate Python Virtual Environment (.venv)
.\.venv\Scripts\Activate.ps1
```
*(On Windows Command Prompt: `.venv\Scripts\activate.bat`)*

### 2. Run Headless CLI Pipeline (Processes Entire Loan Package)
```powershell
python run.py --input Input/downloaded_documents --output Output/processed_results
```
This single command:
1. Ingests all PDF documents in the loan file.
2. Identifies and tags each document across the 16 categories.
3. Extracts borrower, collateral, and loan terms with source page numbers.
4. Calculates As-Is LTV, ARV LTV, LTC, and Borrower Equity.
5. Runs the Random Forest decision model to output `APPROVE`, `CONDITIONAL APPROVAL`, or `DECLINE`.
6. Generates a formal **PDF Underwriting Report** and **JSON Report** in `Output/reports/` and `Output/processed_results/`.

### 3. Run Interactive Web Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to interact with the full underwriting workspace.

---

## Project Structure

```
Application/
├── .venv/                          # Pre-configured Python 3.13 virtual environment
├── Input/
│   ├── downloaded_documents/       # Core loan documents (Term sheet, Appraisal, etc.)
│   └── sample_loan_cases/          # Scenario test cases (JSON)
├── Output/
│   ├── processed_results/          # Extracted JSON results and underwriting summary
│   └── reports/                    # Formatted PDF and JSON reports
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py           # PyMuPDF ingestion with OCR fallback
│   │   └── document_generator.py   # Loan document generator
│   ├── classification/
│   │   ├── document_classifier.py  # Hybrid keyword + llama3.1 classifier
│   │   └── ollama_client.py        # Local Ollama client wrapper
│   ├── extraction/
│   │   └── information_extractor.py# Structured JSON fact extraction with page refs
│   ├── validation/
│   │   ├── financial_calculator.py # Pure Python financial math (LTV, ARV LTV, LTC)
│   │   └── document_validator.py   # Cross-document consistency & policy checks
│   ├── decision/
│   │   └── loan_decision_model.py  # Scikit-Learn Random Forest credit decision model
│   ├── reporting/
│   │   ├── report_generator.py     # ReportLab PDF + JSON export engine
│   │   └── evaluation.py           # Accuracy, precision, recall, F1 computation
│   └── pipeline.py                 # End-to-end pipeline orchestrator
├── models/
│   └── loan_decision_rf.pkl        # Serialized Random Forest classifier
├── app.py                          # Streamlit web user interface
├── run.py                          # Single-command CLI runner
├── WRITEUP.md                      # Comprehensive 3-page project write-up
├── requirements.txt                # Full Python dependencies
└── .env.example                    # Environment variable configuration
```

---

## Core Capabilities

### 1. Document Tagging (16 Categories)
Automatically identifies documents using the Appendix naming convention:
`term_sheet`, `loan_application`, `purchase_contract`, `appraisal`, `scope_of_work`, `title_commitment`, `insurance_binder`, `flood_certificate`, `credit_report`, `bank_statement`, `operating_agreement`, `good_standing`, `government_id`, `track_record`, `background_check`, `other`. Flags missing required documents.

### 2. Information Extraction with Page References
Extracts all key loan parameters without hallucination:
- **Borrower Facts:** Name, entity type, middle credit score (FICO), verified liquid reserves, years of experience, completed flip projects.
- **Loan Facts:** Note amount, purchase price, as-is property value, ARV, renovation budget, interest rate, term length, subject address.
- **Collateral & Due Diligence:** Hazard/builder's risk insurance coverage, FEMA flood zone (Zone X), title encumbrances.
- *Every value retains exact document and page-level attribution.*

### 3. Deterministic Financial Calculations
All financial ratios are computed strictly using deterministic Python:
- **As-Is LTV:** $\frac{\text{Loan Amount}}{\text{As-Is Value}} \times 100$
- **ARV LTV:** $\frac{\text{Loan Amount}}{\text{After Repair Value}} \times 100$
- **LTC:** $\frac{\text{Loan Amount}}{\text{Purchase Price} + \text{Rehab Budget}} \times 100$
- **Borrower Equity:** $\text{Total Project Cost} - \text{Loan Amount}$

### 4. ML Decision Engine
Scikit-Learn Random Forest model evaluates borrower creditworthiness, collateral coverage, liquidity ratios, and document completeness to deliver:
- Recommendation: `APPROVE`, `CONDITIONAL APPROVAL`, or `DECLINE`.
- Calibrated probability distribution across all three decision classes.
- Top influential factors by feature importance.
- Natural-language underwriting narrative generated by `llama3.1`.
- Human review requirements checklist.

---

## Submission Deliverables

- **CLI Runner:** [`run.py`](file:///d:/WORK/resumes/FinacPlus/Application/run.py) processes any loan folder with one command.
- **Web Interface:** [`app.py`](file:///d:/WORK/resumes/FinacPlus/Application/app.py) provides an intuitive 6-tab underwriter dashboard.
- **Evaluation Report:** [`Output/evaluation/EVALUATION_REPORT.md`](file:///d:/WORK/resumes/FinacPlus/Application/Output/evaluation/EVALUATION_REPORT.md) benchmarks 10 test scenarios.
- **Project Write-Up:** [`WRITEUP.md`](file:///d:/WORK/resumes/FinacPlus/Application/WRITEUP.md) covers approach, answers to underwriting questions, trade-offs, AI assistant notes (Google Antigravity), and dataset provenance.
- **Test Suite:** [`tests/test_pipeline.py`](file:///d:/WORK/resumes/FinacPlus/Application/tests/test_pipeline.py) runs 23 unit tests.
