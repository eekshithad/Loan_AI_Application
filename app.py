"""
AI-Driven Loan Underwriting Application
Main Streamlit entry point.
"""
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ── Project root setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # Ensure relative paths work

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Loan Underwriting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
    .metric-card {
        background: white;
        border: 1px solid #e0e6ed;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    .decision-approve { color: #2ecc71; font-size: 24px; font-weight: bold; }
    .decision-conditional { color: #f39c12; font-size: 24px; font-weight: bold; }
    .decision-decline { color: #e74c3c; font-size: 24px; font-weight: bold; }
    .flag-warning { color: #f39c12; }
    .flag-error { color: #e74c3c; }
    .source-tag { font-size: 11px; color: #666; font-style: italic; }
    .disclaimer-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 10px 16px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Initialize and cache the underwriting pipeline."""
    from src.pipeline import UnderwritingPipeline
    return UnderwritingPipeline()


@st.cache_resource(show_spinner=False)
def get_decision_model():
    """Get the trained decision model."""
    from src.decision.loan_decision_model import LoanDecisionModel
    m = LoanDecisionModel()
    m.load_or_train()
    return m


def fmt_currency(val):
    if val is None or val == "Not Found" or val == "N/A":
        return "Not Found"
    try:
        return f"${float(str(val).replace('$', '').replace(',', '')):,.0f}"
    except (ValueError, TypeError):
        return str(val)


def source_tag(source_str: str) -> str:
    return f'<span class="source-tag">📄 {source_str}</span>'


def decision_badge(decision: str) -> str:
    colors = {
        "APPROVE": "#2ecc71",
        "CONDITIONAL_APPROVAL": "#f39c12",
        "DECLINE": "#e74c3c",
    }
    labels = {
        "APPROVE": "✅ APPROVE",
        "CONDITIONAL_APPROVAL": "⚡ CONDITIONAL APPROVAL",
        "DECLINE": "❌ DECLINE",
    }
    color = colors.get(decision, "#666")
    label = labels.get(decision, decision)
    return (
        f'<div style="background:{color}20; border:2px solid {color}; '
        f'border-radius:8px; padding:12px 20px; text-align:center; '
        f'color:{color}; font-size:22px; font-weight:bold;">{label}</div>'
    )


def severity_icon(severity: str) -> str:
    return {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=64)
        st.title("AI Loan Underwriting")
        st.caption("v1.0.0 | FinacPlus Assignment")
        st.divider()

        # Ollama status
        st.subheader("🤖 LLM Status")
        pipeline = get_pipeline()
        ok, msg = pipeline.check_ollama()
        if ok:
            st.success("✅ Ollama: llama3.1 ready")
        else:
            st.error(f"❌ Ollama unavailable\n{msg}")
            st.caption("Using keyword-only classification")

        st.divider()
        st.subheader("📋 Navigation")
        st.caption(
            "Use the tabs above to navigate:\n"
            "• Home\n"
            "• Upload Documents\n"
            "• Analysis Results\n"
            "• Loan Recommendation\n"
            "• Input Documents & Sources\n"
            "• Evaluation"
        )

        st.divider()
        st.caption("FinacPlus Commercial Real Estate Lending System")


# ── Tab: Home ─────────────────────────────────────────────────────────────────

def tab_home():
    st.title("🏦 AI-Driven Loan Underwriting System")
    st.markdown(
        "An intelligent assistant for real estate bridge loan underwriting. "
        "Upload loan documents, and the system will classify, extract, analyze, "
        "and generate a loan recommendation."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document Categories", "16")
    with col2:
        st.metric("ML Model", "Random Forest")
    with col3:
        st.metric("LLM", "llama3.1 (local)")
    with col4:
        st.metric("Report Formats", "PDF + JSON")

    st.divider()
    st.subheader("📋 How It Works")

    steps = [
        ("1️⃣ Upload", "Upload one or multiple PDF documents from a loan file."),
        ("2️⃣ Classify", "AI identifies each document type (Term Sheet, Appraisal, Credit Report, etc.)"),
        ("3️⃣ Extract", "LLM extracts borrower info, loan details, and financial figures with page references."),
        ("4️⃣ Calculate", "Python calculates LTV, ARV LTV, LTC, and borrower equity from extracted data."),
        ("5️⃣ Validate", "Checks for missing documents, conflicting values, and policy violations."),
        ("6️⃣ Decide", "Random Forest ML model generates APPROVE / CONDITIONAL APPROVAL / DECLINE."),
        ("7️⃣ Report", "Full underwriting report generated as PDF and JSON for download."),
    ]

    for step, desc in steps:
        col_a, col_b = st.columns([1, 5])
        with col_a:
            st.markdown(f"**{step}**")
        with col_b:
            st.write(desc)

    st.divider()
    st.subheader("🏗️ Architecture")

    st.markdown("""
    ```
    PDF Upload
        │
        ▼
    PDF Ingestion (PyMuPDF + OCR)
        │
        ▼
    Document Classification (Keyword Heuristics + llama3.1)
        │
        ▼
    Information Extraction (llama3.1 → structured JSON)
        │
        ├──► Financial Calculations (Python — deterministic)
        │         As-Is LTV, ARV LTV, LTC, Equity
        │
        ├──► Document Validation (Python — rule-based)
        │         Missing docs, conflicts, policy checks
        │
        └──► ML Decision Engine (Random Forest / scikit-learn)
                  APPROVE / CONDITIONAL_APPROVAL / DECLINE
                       │
                       ▼
              LLM Explanation (llama3.1)
                       │
                       ▼
              Underwriting Report (PDF + JSON)
    ```
    """)


# ── Tab: Upload ───────────────────────────────────────────────────────────────

def tab_upload():
    st.title("📤 Upload Loan Documents")

    st.info(
        "Upload PDF loan documents. The system supports multiple files and "
        "combined PDFs. Documents will be classified, and information extracted automatically."
    )

    # Option to use pre-loaded sample documents
    st.subheader("Option A — Use Sample Documents")
    if st.button("📂 Load Pre-Generated Sample Documents", type="primary"):
        with st.spinner("Generating sample documents…"):
            try:
                from src.ingestion.document_generator import generate_all_documents
                docs = generate_all_documents()
                st.success(f"✅ Generated {len(docs)} sample documents in Input/downloaded_documents/")
                st.session_state["sample_docs"] = docs
            except Exception as exc:
                st.error(f"Error generating documents: {exc}")

    if st.button("🚀 Process Sample Documents"):
        docs = st.session_state.get("sample_docs")
        if not docs:
            # Check both Input/downloaded_documents and Input directly
            found_paths = list((ROOT / "Input" / "downloaded_documents").glob("*.pdf"))
            found_paths += list((ROOT / "Input").glob("*.pdf"))
            # Deduplicate by filename
            doc_map = {p.name: str(p) for p in found_paths}
            docs = sorted(list(doc_map.values()))
        if not docs:
            st.warning("No sample documents found. Click 'Load Pre-Generated Sample Documents' first.")
        else:
            _run_pipeline(docs)

    st.divider()
    st.subheader("Option B — Upload Your Own Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files from a loan package",
    )

    if uploaded_files and st.button("🔍 Analyze Uploaded Documents", type="primary"):
        # Save uploaded files to temp directory
        tmp_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded_files:
                tmp_path = Path(tmpdir) / uf.name
                with open(tmp_path, "wb") as f:
                    f.write(uf.read())
                tmp_paths.append(str(tmp_path))
            _run_pipeline(tmp_paths)


def _run_pipeline(file_paths: list[str]):
    """Execute the full pipeline and store results in session state."""
    pipeline = get_pipeline()

    progress_bar = st.progress(0, text="Starting analysis…")

    def update_progress(step: str, pct: int):
        progress_bar.progress(pct / 100, text=step)

    with st.spinner("Processing documents…"):
        try:
            results = pipeline.process_files(
                file_paths,
                save_json=True,
                save_pdf=True,
                progress_callback=update_progress,
            )
            st.session_state["pipeline_results"] = results
            progress_bar.progress(1.0, text="✅ Analysis complete!")
            if results.get("errors"):
                for err in results["errors"]:
                    st.warning(err)
            st.success("✅ Analysis complete! Navigate to the Analysis Results tab.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            logger.exception("Pipeline failed")


# ── Tab: Analysis Results ─────────────────────────────────────────────────────

def tab_analysis():
    st.title("📊 Analysis Results")

    results = st.session_state.get("pipeline_results")
    if not results:
        st.info("No analysis results yet. Upload and process documents in the Upload tab.")
        return

    report = results.get("report", {})
    if not report:
        st.error("Analysis failed — no report generated.")
        return

    # ── Document Classification ──
    st.subheader("🗂️ Document Classification")
    dc = report.get("document_classification", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Uploaded", len(dc.get("uploaded_files", [])))
    with col2:
        st.metric("Missing Required", len(dc.get("missing_required", [])))
    with col3:
        st.metric("Completeness", f"{dc.get('completeness_pct', 0):.0f}%")

    classified = dc.get("classified", [])
    if classified:
        df_cls = pd.DataFrame(classified)
        df_cls.columns = ["File", "Category", "Confidence", "Method"]
        st.dataframe(df_cls, use_container_width=True, hide_index=True)

    if dc.get("missing_required"):
        st.error("**Missing Required Documents:**")
        for m in dc["missing_required"]:
            st.markdown(f"• ❌ {m}")

    if dc.get("missing_recommended"):
        with st.expander("⚠️ Missing Recommended Documents"):
            for m in dc["missing_recommended"]:
                st.markdown(f"• {m}")

    st.divider()

    # ── Extracted Information ──
    st.subheader("📋 Extracted Information")

    col_b, col_l = st.columns(2)

    with col_b:
        st.markdown("**Borrower Information**")
        borrower = report.get("extracted_information", {}).get("borrower", {})
        b_rows = []
        labels = {
            "name": "Borrower Name",
            "entity_type": "Entity Type",
            "credit_score": "Credit Score (FICO)",
            "employment": "Employment / Business",
            "bank_balance": "Bank Balance",
            "experience_years": "Experience (Years)",
            "projects_completed": "Projects Completed",
        }
        for key, label in labels.items():
            if key in borrower:
                field_data = borrower[key]
                b_rows.append({
                    "Field": label,
                    "Value": field_data.get("value", "Not Found"),
                    "Source": field_data.get("source", "N/A"),
                })
        if b_rows:
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

    with col_l:
        st.markdown("**Loan Information**")
        loan = report.get("extracted_information", {}).get("loan", {})
        l_rows = []
        labels_l = {
            "loan_amount": "Loan Amount",
            "purchase_price": "Purchase Price",
            "as_is_value": "As-Is Property Value",
            "arv": "After Repair Value (ARV)",
            "rehab_budget": "Rehabilitation Budget",
            "property_address": "Property Address",
            "interest_rate": "Interest Rate",
            "loan_term_months": "Loan Term",
            "origination_fee": "Origination Fee",
        }
        for key, label in labels_l.items():
            if key in loan:
                field_data = loan[key]
                l_rows.append({
                    "Field": label,
                    "Value": field_data.get("value", "Not Found"),
                    "Source": field_data.get("source", "N/A"),
                })
        if l_rows:
            st.dataframe(pd.DataFrame(l_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Financial Calculations ──
    st.subheader("🧮 Financial Calculations")
    fm = report.get("financial_metrics", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_map = [
        (col1, "As-Is LTV", fm.get("as_is_ltv", "N/A"), "≤ 80%"),
        (col2, "ARV LTV", fm.get("arv_ltv", "N/A"), "≤ 75%"),
        (col3, "LTC", fm.get("ltc", "N/A"), "≤ 90%"),
        (col4, "Total Project Cost", fm.get("total_project_cost", "N/A"), "—"),
        (col5, "Borrower Equity", fm.get("borrower_equity", "N/A"), "> $0"),
    ]
    for col, label, value, threshold in metrics_map:
        with col:
            st.metric(label, str(value), help=f"Threshold: {threshold}")

    if fm.get("risk_flags"):
        st.markdown("**Risk Flags:**")
        for flag in fm["risk_flags"]:
            if "🚨" in flag:
                st.error(flag)
            elif "⚠️" in flag:
                st.warning(flag)
            else:
                st.info(flag)

    st.divider()

    # ── Validation Issues ──
    st.subheader("🔍 Validation & Consistency Issues")
    issues = report.get("validation_issues", [])
    if not issues:
        st.success("✅ No significant validation issues found.")
    else:
        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") == "warning"]
        infos = [i for i in issues if i.get("severity") == "info"]

        if errors:
            st.error(f"**{len(errors)} Critical Issue(s):**")
            for issue in errors:
                with st.expander(f"🔴 {issue.get('code', '')} – {issue.get('description', '')[:60]}"):
                    st.write(f"**Description:** {issue.get('description', '')}")
                    st.write(f"**Action Required:** {issue.get('action', '')}")

        if warnings:
            st.warning(f"**{len(warnings)} Warning(s):**")
            for issue in warnings:
                with st.expander(f"🟡 {issue.get('code', '')} – {issue.get('description', '')[:60]}"):
                    st.write(f"**Description:** {issue.get('description', '')}")
                    st.write(f"**Suggested Action:** {issue.get('action', '')}")


# ── Tab: Loan Recommendation ──────────────────────────────────────────────────

def tab_recommendation():
    st.title("🎯 Loan Recommendation")

    results = st.session_state.get("pipeline_results")
    if not results:
        st.info("No analysis results yet. Upload and process documents in the Upload tab.")
        return

    report = results.get("report", {})
    rec = report.get("loan_recommendation", {})

    if not rec:
        st.error("No recommendation generated.")
        return

    decision = rec.get("decision", "N/A")
    st.markdown(decision_badge(decision), unsafe_allow_html=True)
    st.markdown(f"**Model Confidence:** {rec.get('confidence', 'N/A')}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Decision Probabilities")
        probs = rec.get("probabilities", {})
        if probs:
            prob_df = pd.DataFrame([
                {"Decision": k.replace("_", " "), "Probability": f"{v:.1%}"}
                for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Simple bar visualization
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 2.5))
            keys = [k.replace("_", " ") for k in probs.keys()]
            vals = [v * 100 for v in probs.values()]
            bar_colors = []
            for k in probs.keys():
                if k == "APPROVE":
                    bar_colors.append("#2ecc71")
                elif k == "CONDITIONAL_APPROVAL":
                    bar_colors.append("#f39c12")
                else:
                    bar_colors.append("#e74c3c")
            ax.barh(keys, vals, color=bar_colors)
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            ax.set_title("Decision Probabilities")
            for i, (k, v) in enumerate(zip(keys, vals)):
                ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.subheader("🔑 Key Decision Factors")
        factors = rec.get("top_factors", [])
        if factors:
            for f in factors:
                name = f.get("feature", "").replace("_", " ").title()
                imp = f.get("importance", 0)
                st.progress(imp, text=f"{name}: {imp:.1%}")

    st.divider()

    st.subheader("📝 Underwriter Explanation")
    explanation = rec.get("explanation", "No explanation generated.")
    st.markdown(explanation)

    st.divider()

    st.subheader("✅ Human Review Requirements")
    for item in report.get("human_review_requirements", []):
        st.markdown(f"☐ {item}")

    st.divider()

    # ── Download Report ──
    st.subheader("⬇️ Download Report")
    col_j, col_p = st.columns(2)

    with col_j:
        json_path = results.get("report_json_path")
        if json_path and Path(json_path).exists():
            with open(json_path, "rb") as f:
                st.download_button(
                    "📥 Download JSON Report",
                    data=f,
                    file_name=Path(json_path).name,
                    mime="application/json",
                    type="primary",
                )
        else:
            if report:
                json_bytes = json.dumps(report, indent=2, default=str).encode()
                st.download_button(
                    "📥 Download JSON Report",
                    data=json_bytes,
                    file_name="underwriting_report.json",
                    mime="application/json",
                )

    with col_p:
        pdf_path = results.get("report_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    data=f,
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    type="primary",
                )


# ── Tab: Input Documents & Sources ────────────────────────────────────────────

def tab_sources():
    st.title("📚 Input Documents & Sources")

    st.markdown("""
    This section provides the complete provenance register for all 12 input documents in the loan file,
    documenting source templates, originators, underwriting rationale, and operational parameters.
    """)
    st.divider()

    try:
        from src.ingestion.document_generator import DOCUMENT_SOURCES
        for src in DOCUMENT_SOURCES:
            with st.expander(f"📄 {src['file_name']} — {src['category']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Category:** {src['category']}")
                    st.markdown(f"**Type:** `{src['source_type']}`")
                    st.markdown(f"**URL:** {src['url']}")
                    st.markdown(f"**Source:** {src['source_name']}")
                with col2:
                    st.markdown(f"**Why Selected:** {src['rationale']}")
                    st.markdown(f"**Limitations:** _{src['limitations']}_")
    except Exception as exc:
        st.error(f"Could not load source information: {exc}")

    st.divider()
    st.subheader("Document Package Overview")
    st.markdown("""
    The loan file complies with standard commercial bridge lending underwriting requirements.
    All documents represent standard institutional formats referenced in the assignment brief.
    """)


# ── Tab: Evaluation ───────────────────────────────────────────────────────────

def tab_evaluation():
    st.title("📈 Model Evaluation")

    st.markdown("""
    Performance benchmarks of the machine learning decision model and document classifier
    measured across 10 structured loan underwriting scenarios.
    """)
    st.divider()

    # ── Loan Decision Evaluation ──
    st.subheader("🏦 Loan Decision Model Performance")

    with st.spinner("Running evaluation on ground-truth cases…"):
        try:
            from src.reporting.evaluation import (
                evaluate_loan_decision_model, GROUND_TRUTH_LOAN_CASES
            )
            model = get_decision_model()
            eval_result = evaluate_loan_decision_model(model)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{eval_result.accuracy:.1%}")
            with col2:
                st.metric("Precision", f"{eval_result.precision:.1%}")
            with col3:
                st.metric("Recall", f"{eval_result.recall:.1%}")
            with col4:
                st.metric("F1 Score", f"{eval_result.f1:.1%}")

            st.caption(eval_result.model_disclosure)

            # Case results
            st.subheader("Test Cases")
            case_df = pd.DataFrame([
                {
                    "Case ID": c["case_id"],
                    "Scenario": c["scenario"],
                    "True Label": c["true_label"],
                    "Predicted": c["predicted_label"],
                    "Confidence": f"{c['confidence']:.1%}",
                    "Correct": "✅" if c["correct"] else "❌",
                    "Notes": c["notes"],
                }
                for c in eval_result.case_results
            ])
            st.dataframe(case_df, use_container_width=True, hide_index=True)

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = eval_result.confusion_matrix
            class_names = ["APPROVE", "CONDITIONAL_APPROVAL", "DECLINE"]
            # Filter to available classes
            report_classes = list(eval_result.per_class_report.keys())
            report_classes = [c for c in class_names if c in report_classes]

            if cm and report_classes:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Rebuild CM with available classes
                from sklearn.metrics import confusion_matrix as sk_cm
                true_l = [c["true_label"] for c in eval_result.case_results]
                pred_l = [c["predicted_label"] for c in eval_result.case_results]
                cm_arr = sk_cm(true_l, pred_l, labels=class_names)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    cm_arr, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[c.replace("_", "\n") for c in class_names],
                    yticklabels=[c.replace("_", "\n") for c in class_names],
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix — Loan Decision Model")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Per-class report
            with st.expander("📊 Detailed Classification Report"):
                for cls in class_names:
                    if cls in eval_result.per_class_report:
                        d = eval_result.per_class_report[cls]
                        if isinstance(d, dict):
                            st.markdown(f"**{cls.replace('_', ' ')}**")
                            col_p, col_r, col_f, col_s = st.columns(4)
                            with col_p:
                                st.metric("Precision", f"{d.get('precision', 0):.1%}")
                            with col_r:
                                st.metric("Recall", f"{d.get('recall', 0):.1%}")
                            with col_f:
                                st.metric("F1", f"{d.get('f1-score', 0):.1%}")
                            with col_s:
                                st.metric("Support", int(d.get('support', 0)))

        except Exception as exc:
            st.error(f"Evaluation error: {exc}")
            logger.exception("Evaluation failed")

    st.divider()

    # ── Training Data Disclosure ──
    st.subheader("📋 Model Training Information")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        **Library:** scikit-learn
        **Training Samples:** 40 structured training cases
        **Test Split:** 25% (10 cases)
        **Features Used:** 8
        """)
    with col_b:
        st.markdown("""
        **Features:**
        1. Credit Score
        2. As-Is LTV
        3. ARV LTV
        4. Loan-to-Cost (LTC)
        5. Bank Balance Ratio
        6. Document Completeness %
        7. Experience Years
        8. Missing Required Documents
        """)

    st.divider()

    # ── Ground Truth Cases ──
    st.subheader("🏷️ Benchmark Underwriting Cases")
    from src.reporting.evaluation import GROUND_TRUTH_LOAN_CASES
    gt_df = pd.DataFrame([
        {
            "Case ID": c["case_id"],
            "Scenario": c["scenario"],
            "Credit Score": c["credit_score"],
            "As-Is LTV": f"{c['as_is_ltv']:.0f}%",
            "ARV LTV": f"{c['arv_ltv']:.0f}%",
            "LTC": f"{c['ltc']:.0f}%",
            "True Label": c["true_label"],
            "Label Source": c["label_source"],
        }
        for c in GROUND_TRUTH_LOAN_CASES
    ])
    st.dataframe(gt_df, use_container_width=True, hide_index=True)
    st.caption("Benchmark scenarios designed across high-risk, borderline, and fully approved borrower profiles.")

    st.divider()
    st.subheader("System Limitations & Operational Boundaries")
    st.markdown("""
    1. **Document Formatting Variations:** Non-standard or handwritten loan documents may require manual verification.
    2. **Local Model Latency:** Ollama CPU inference requires ~15–30 seconds per document without GPU acceleration.
    3. **Threshold Sensitivity:** Frontier borrowers near policy boundaries (e.g., FICO 640–650) are routed for human underwriter signoff.
    """)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    tabs = st.tabs([
        "🏠 Home",
        "📤 Upload Documents",
        "📊 Analysis Results",
        "🎯 Loan Recommendation",
        "📚 Input Documents & Sources",
        "📈 Evaluation",
    ])

    with tabs[0]:
        tab_home()
    with tabs[1]:
        tab_upload()
    with tabs[2]:
        tab_analysis()
    with tabs[3]:
        tab_recommendation()
    with tabs[4]:
        tab_sources()
    with tabs[5]:
        tab_evaluation()


if __name__ == "__main__":
    main()

