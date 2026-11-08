"""
AI-Driven Loan Underwriting Application
A clean, simple assistant for real estate bridge loan underwriting.
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

# Project root setup
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(
    page_title="Loan Underwriting Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Clean Styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    h1, h2, h3 { color: #1a3c5e; }
    .badge-approve { background:#d4edda; color:#155724; padding:8px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
    .badge-conditional { background:#fff3cd; color:#856404; padding:8px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
    .badge-decline { background:#f8d7da; color:#721c24; padding:8px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import UnderwritingPipeline
    return UnderwritingPipeline()


@st.cache_resource(show_spinner=False)
def get_decision_model():
    from src.decision.loan_decision_model import LoanDecisionModel
    m = LoanDecisionModel()
    m.load_or_train()
    return m


def load_default_results() -> bool:
    """Pre-load existing analysis results so the app is immediately ready without waiting."""
    summary_path = ROOT / "Output" / "processed_results" / "underwriting_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pdf_files = sorted(list((ROOT / "Output" / "reports").glob("*.pdf")), key=lambda p: p.stat().st_mtime, reverse=True)
            pdf_path = str(pdf_files[0]) if pdf_files else None
            st.session_state["pipeline_results"] = {
                "report": data,
                "report_json_path": str(summary_path),
                "report_pdf_path": pdf_path,
            }
            return True
        except Exception as e:
            logger.warning(f"Could not load pre-analyzed summary: {e}")
    return False


def decision_badge(decision: str):
    if decision == "APPROVE":
        return '<div class="badge-approve">✅ RECOMMENDATION: APPROVE</div>'
    elif decision == "CONDITIONAL_APPROVAL":
        return '<div class="badge-conditional">⚡ RECOMMENDATION: CONDITIONAL APPROVAL</div>'
    return '<div class="badge-decline">❌ RECOMMENDATION: DECLINE</div>'


def _run_pipeline(file_paths: list[str], fast_mode: bool = False):
    pipeline = get_pipeline()
    bar = st.progress(0, text="Initializing underwriting analysis...")

    def update_p(step, pct):
        bar.progress(pct / 100, text=step)

    try:
        results = pipeline.process_files(
            file_paths,
            save_json=True,
            save_pdf=True,
            progress_callback=update_p,
            fast_mode=fast_mode,
        )
        st.session_state["pipeline_results"] = results
        bar.progress(1.0, text="Completed!")
        st.success("Analysis complete! Check the Analysis Results and Recommendation tabs.")
    except Exception as exc:
        st.error(f"Processing error: {exc}")
        logger.exception("Pipeline run failed")


# ── TAB 1: HOME ───────────────────────────────────────────────────────────────
def tab_home():
    st.title("🏦 AI Loan Underwriting Assistant")
    st.write(
        "An automated decision-support tool that reads loan packages, verifies property & borrower facts, "
        "computes credit risk metrics, and provides an explainable underwriting recommendation."
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Core Documents", "4 Primary PDFs")
    with col2:
        st.metric("Decision Engine", "Random Forest ML")
    with col3:
        pipeline = get_pipeline()
        ok, msg = pipeline.check_ollama()
        st.metric("Ollama LLM", "llama3.1 Active" if ok else "Offline")

    st.divider()
    st.subheader("How It Works")
    st.markdown("""
    1. **Upload:** Provide the borrower's loan package (Term Sheet, Application, Contract, Appraisal).
    2. **Classify & Extract:** System identifies document types and extracts key figures with page citations.
    3. **Calculate:** Pure Python calculates As-Is LTV, ARV LTV, LTC, and Borrower Equity.
    4. **Recommend:** Machine learning model evaluates credit risk and produces a justified decision report.
    """)


# ── TAB 2: UPLOAD ─────────────────────────────────────────────────────────────
def tab_upload():
    st.title("📤 Loan Documents")

    st.markdown("### 📥 Download Sample Documents")
    st.write("Click any button below to download the sample loan documents, then upload them into the uploader below to test:")

    sample_dir = ROOT / "Input" / "downloaded_documents"
    if not sample_dir.exists():
        sample_dir = ROOT / "Input"

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    file_specs = [
        ("term_sheet_sample.pdf", "📄 Term Sheet", col_d1),
        ("loan_application_sample.pdf", "📄 Loan Application", col_d2),
        ("purchase_contract_sample.pdf", "📄 Purchase Contract", col_d3),
        ("appraisal_sample.pdf", "📄 Appraisal Report", col_d4),
    ]

    for fname, label, col in file_specs:
        fpath = sample_dir / fname
        with col:
            if fpath.exists():
                with open(fpath, "rb") as fp:
                    st.download_button(
                        label=label,
                        data=fp.read(),
                        file_name=fname,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.caption(f"{fname} missing")

    st.caption("💡 Local directory containing all sample PDFs: `D:\\WORK\\resumes\\FinacPlus\\Application\\Input\\downloaded_documents\\`")
    st.divider()

    # CPU Optimization Toggle
    fast_mode = st.checkbox(
        "⚡ Fast Mode (CPU-Optimized Heuristics — Recommended for CPU)",
        value=True,
        help="Runs deterministic heuristic classification & extraction in ~1-2 seconds on CPU, avoiding long LLM generation times."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Default Loan Package")
        st.caption("Standard commercial bridge loan package (4 core files).")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🚀 Process Loan Package", type="primary", use_container_width=True):
                docs = sorted([str(p) for p in (ROOT / "Input" / "downloaded_documents").glob("*.pdf")])
                if not docs:
                    docs = sorted([str(p) for p in (ROOT / "Input").glob("*.pdf") if p.is_file()])
                if docs:
                    _run_pipeline(docs, fast_mode=fast_mode)
                else:
                    st.warning("No PDF documents found in Input/ directory.")
        with b2:
            if st.button("⚡ Quick Load Pre-Analyzed", use_container_width=True):
                if load_default_results():
                    st.success("Loaded pre-analyzed loan package results instantly!")
                else:
                    st.warning("No pre-computed results found.")

    with col2:
        st.subheader("Upload Custom PDFs")
        uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if uploaded and st.button("🔍 Analyze Uploaded Files", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for uf in uploaded:
                    p = Path(tmpdir) / uf.name
                    with open(p, "wb") as f:
                        f.write(uf.read())
                    paths.append(str(p))
                _run_pipeline(paths, fast_mode=fast_mode)


# ── TAB 3: RESULTS ────────────────────────────────────────────────────────────
def tab_results():
    st.title("📊 Analysis Results")

    res = st.session_state.get("pipeline_results")
    if not res or not res.get("report"):
        st.info("No documents analyzed yet. Click 'Process Loan Package' in the Upload tab.")
        return

    report = res["report"]
    fm = report.get("financial_metrics", {})
    dc = report.get("document_classification", {})
    ext = report.get("extracted_information", {})

    # Key Metrics Header
    st.subheader("Financial Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("As-Is LTV", fm.get("as_is_ltv", "N/A"), help="Target: ≤ 80%")
    c2.metric("ARV LTV", fm.get("arv_ltv", "N/A"), help="Target: ≤ 75%")
    c3.metric("LTC", fm.get("ltc", "N/A"), help="Target: ≤ 90%")
    c4.metric("Borrower Equity", fm.get("borrower_equity", "N/A"))

    if fm.get("risk_flags"):
        st.write("")
        for f in fm["risk_flags"]:
            st.warning(f)

    st.divider()

    # Document Classification & Facts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Document Classification")
        cls_list = dc.get("classified", [])
        if cls_list:
            df_c = pd.DataFrame(cls_list)[["file", "category", "confidence"]]
            df_c.columns = ["Document Name", "Identified Category", "Confidence"]
            st.dataframe(df_c, use_container_width=True, hide_index=True)

        if dc.get("missing_required"):
            st.error(f"Missing Required Documents: {', '.join(dc['missing_required'])}")

    with col_b:
        st.subheader("Key Extracted Facts")
        loan_info = ext.get("loan", {})
        borrower_info = ext.get("borrower", {})

        facts = [
            {"Parameter": "Borrower Entity", "Value": borrower_info.get("name", {}).get("value", "N/A"), "Source": borrower_info.get("name", {}).get("source", "N/A")},
            {"Parameter": "Credit Score (FICO)", "Value": borrower_info.get("credit_score", {}).get("value", "N/A"), "Source": borrower_info.get("credit_score", {}).get("source", "N/A")},
            {"Parameter": "Loan Amount", "Value": loan_info.get("loan_amount", {}).get("value", "N/A"), "Source": loan_info.get("loan_amount", {}).get("source", "N/A")},
            {"Parameter": "Purchase Price", "Value": loan_info.get("purchase_price", {}).get("value", "N/A"), "Source": loan_info.get("purchase_price", {}).get("source", "N/A")},
            {"Parameter": "As-Is Property Value", "Value": loan_info.get("as_is_value", {}).get("value", "N/A"), "Source": loan_info.get("as_is_value", {}).get("source", "N/A")},
            {"Parameter": "After Repair Value (ARV)", "Value": loan_info.get("arv", {}).get("value", "N/A"), "Source": loan_info.get("arv", {}).get("source", "N/A")},
        ]
        st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)


# ── TAB 4: RECOMMENDATION ─────────────────────────────────────────────────────
def tab_recommendation():
    st.title("🎯 Loan Recommendation")

    res = st.session_state.get("pipeline_results")
    if not res or not res.get("report"):
        st.info("Run document processing first to see the recommendation.")
        return

    report = res["report"]
    rec = report.get("loan_recommendation", {})
    decision = rec.get("decision", "DECLINE")

    st.markdown(decision_badge(decision), unsafe_allow_html=True)
    st.caption(f"Model Confidence: {rec.get('confidence', 'N/A')}")
    st.write("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Decision Probabilities")
        probs = rec.get("probabilities", {})
        if probs:
            p_data = [{"Outcome": k.replace("_", " "), "Probability": f"{v:.1%}"} for k, v in probs.items()]
            st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Top Decision Factors")
        factors = rec.get("top_factors", [])
        if factors:
            for f in factors[:4]:
                name = f["feature"].replace("_", " ").title()
                st.progress(f["importance"], text=f"{name}: {f['importance']:.1%}")

    st.divider()
    st.subheader("Underwriting Explanation")
    st.write(rec.get("explanation", "Decision based on collateral and credit criteria."))

    st.divider()
    st.subheader("Download Full Underwriting Report")
    c_pdf, c_json = st.columns(2)

    with c_pdf:
        pdf_path = res.get("report_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    data=f,
                    file_name="underwriting_report.pdf",
                    mime="application/pdf",
                    type="primary",
                )

    with c_json:
        json_path = res.get("report_json_path")
        if json_path and Path(json_path).exists():
            with open(json_path, "rb") as f:
                st.download_button(
                    "📥 Download JSON Report",
                    data=f,
                    file_name="underwriting_report.json",
                    mime="application/json",
                )


# ── TAB 5: SOURCES ────────────────────────────────────────────────────────────
def tab_sources():
    st.title("📚 Input Documents & Sources")
    st.write("Specification and provenance for the core underwriting documents in `Input/`.")
    st.write("")

    from src.ingestion.document_generator import DOCUMENT_SOURCES
    src_data = [
        {
            "Document": s["file_name"],
            "Category": s["category"],
            "Source Template": s["source_name"],
            "Underwriting Purpose": s["rationale"],
        }
        for s in DOCUMENT_SOURCES
    ]
    st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)


# ── TAB 6: EVALUATION ─────────────────────────────────────────────────────────
def tab_evaluation():
    st.title("📈 Model Evaluation")
    st.write("Benchmark performance of the decision model across 10 structured underwriting test cases.")
    st.write("")

    from src.reporting.evaluation import evaluate_loan_decision_model
    model = get_decision_model()
    eval_res = evaluate_loan_decision_model(model)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{eval_res.accuracy:.1%}")
    c2.metric("Precision", f"{eval_res.precision:.1%}")
    c3.metric("Recall", f"{eval_res.recall:.1%}")
    c4.metric("F1 Score", f"{eval_res.f1:.1%}")

    st.divider()
    st.subheader("Benchmark Scenario Results (`GROUND_TRUTH_LOAN_CASES`)")
    table_rows = []
    for cr in eval_res.case_results:
        table_rows.append({
            "Case ID": cr["case_id"],
            "Scenario": cr["scenario"],
            "Expected Decision": cr["true_label"],
            "Model Prediction": cr["predicted_label"],
            "Confidence": f"{cr['confidence']:.1%}",
            "Result": "✅ Match" if cr["correct"] else "❌ Mismatch",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Sample Loan Scenarios (`Input/sample_loan_cases/`)")

    cases_dir = ROOT / "Input" / "sample_loan_cases"
    case_rows = []
    for f in sorted(cases_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as jf:
            cd = json.load(jf)
            case_rows.append({
                "Case": cd.get("case_id"),
                "Scenario": cd.get("title"),
                "Borrower Entity": cd.get("borrower_entity"),
                "Credit Score": cd.get("credit_score"),
                "Expected Decision": cd.get("expected_decision"),
            })

    if case_rows:
        st.dataframe(pd.DataFrame(case_rows), use_container_width=True, hide_index=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # Pre-populate session state so Analysis & Recommendation tabs are ready immediately
    if "pipeline_results" not in st.session_state:
        load_default_results()

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
        tab_results()
    with tabs[3]:
        tab_recommendation()
    with tabs[4]:
        tab_sources()
    with tabs[5]:
        tab_evaluation()


if __name__ == "__main__":
    main()
