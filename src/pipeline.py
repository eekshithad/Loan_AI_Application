"""
Pipeline Orchestrator
Ties together all modules into a single end-to-end workflow.
"""
import logging
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_loader import load_pdf, PDFDocument
from src.classification.document_classifier import (
    classify_document, identify_missing_categories
)
from src.classification.ollama_client import OllamaClient
from src.extraction.information_extractor import (
    extract_from_document, merge_extractions
)
from src.validation.financial_calculator import calculate_metrics
from src.validation.document_validator import validate_documents
from src.decision.loan_decision_model import (
    LoanDecisionModel, build_feature_vector, generate_decision_explanation
)
from src.reporting.report_generator import (
    build_report, save_report_json, save_report_pdf
)

logger = logging.getLogger(__name__)


class UnderwritingPipeline:
    """
    End-to-end loan underwriting pipeline.
    Call process_files() with a list of PDF paths to get a full report.
    """

    def __init__(self, ollama_model: str = "llama3.1",
                 ollama_base_url: str = "http://localhost:11434"):
        self.ollama = OllamaClient(model=ollama_model, base_url=ollama_base_url)
        self.decision_model = LoanDecisionModel()
        self.decision_model.load_or_train()
        self._ollama_ok = None

    def check_ollama(self) -> tuple[bool, str]:
        if self._ollama_ok is None:
            self._ollama_ok = self.ollama.is_available()
        return self._ollama_ok

    def process_files(
        self,
        file_paths: list[str],
        save_json: bool = True,
        save_pdf: bool = True,
        progress_callback=None,
    ) -> dict:
        """
        Main processing entry point.
        Returns a dict with all pipeline results.
        """
        def progress(step: str, pct: int):
            logger.info(f"[{pct}%] {step}")
            if progress_callback:
                progress_callback(step, pct)

        results = {
            "status": "processing",
            "file_paths": file_paths,
            "loaded_docs": [],
            "classified_docs": [],
            "extraction_result": None,
            "financial_metrics": None,
            "validation_result": None,
            "prediction": None,
            "explanation": "",
            "report": None,
            "report_json_path": None,
            "report_pdf_path": None,
            "errors": [],
        }

        ollama_ok, ollama_msg = self.check_ollama()
        if not ollama_ok:
            logger.warning(f"Ollama unavailable: {ollama_msg}. Using keyword-only classification.")
            results["errors"].append(f"⚠️ Ollama: {ollama_msg}")

        # ── Step 1: Load PDFs ──
        progress("Loading PDF documents…", 5)
        for fp in file_paths:
            doc = load_pdf(fp)
            if doc.error:
                results["errors"].append(f"Load error: {doc.file_name}: {doc.error}")
            results["loaded_docs"].append(doc)

        if not results["loaded_docs"]:
            results["status"] = "error"
            results["errors"].append("No documents could be loaded.")
            return results

        # ── Step 2: Classify ──
        progress("Classifying documents…", 20)
        for doc in results["loaded_docs"]:
            classification = classify_document(
                text=doc.full_text,
                file_name=doc.file_name,
                ollama_client=self.ollama if ollama_ok else None,
                use_llm=ollama_ok,
            )
            results["classified_docs"].append(classification)

        # ── Step 3: Extract ──
        progress("Extracting information…", 40)
        raw_extractions = []
        for doc, cls in zip(results["loaded_docs"], results["classified_docs"]):
            extracted = extract_from_document(doc, cls.category, self.ollama if ollama_ok else _dummy_ollama())
            raw_extractions.append((extracted, doc))

        extraction_result = merge_extractions(raw_extractions)
        results["extraction_result"] = extraction_result

        # ── Step 4: Financial Calculations ──
        progress("Calculating financial metrics…", 55)
        from src.extraction.information_extractor import NOT_FOUND

        def to_float(field_obj):
            if field_obj is None:
                return None
            val = getattr(field_obj, 'value', None)
            if val == NOT_FOUND or val is None:
                return None
            try:
                return float(str(val).replace('$', '').replace(',', ''))
            except (ValueError, TypeError):
                return None

        fm = calculate_metrics(
            loan_amount=to_float(extraction_result.loan.loan_amount),
            purchase_price=to_float(extraction_result.loan.purchase_price),
            as_is_value=to_float(extraction_result.loan.as_is_value),
            arv=to_float(extraction_result.loan.arv),
            rehab_budget=to_float(extraction_result.loan.rehab_budget),
        )
        results["financial_metrics"] = fm

        # ── Step 5: Validation ──
        progress("Validating documents and data…", 65)
        classified_categories = [c.category for c in results["classified_docs"]]
        val_result = validate_documents(classified_categories, extraction_result, fm)
        results["validation_result"] = val_result

        # ── Step 6: ML Decision ──
        progress("Running ML decision model…", 78)
        feature_vec = build_feature_vector(
            credit_score=to_float(extraction_result.borrower.credit_score),
            as_is_ltv=fm.as_is_ltv,
            arv_ltv=fm.arv_ltv,
            ltc=fm.ltc,
            bank_balance=to_float(extraction_result.borrower.bank_balance),
            loan_amount=to_float(extraction_result.loan.loan_amount),
            doc_completeness_pct=val_result.overall_completeness_pct,
            experience_years=to_float(extraction_result.borrower.experience_years),
            missing_required_docs=len(val_result.missing_required),
        )
        prediction = self.decision_model.predict(feature_vec)
        results["prediction"] = prediction

        # ── Step 7: LLM Explanation ──
        progress("Generating underwriter explanation…", 88)
        explanation = generate_decision_explanation(
            prediction, extraction_result, fm, val_result,
            self.ollama if ollama_ok else _DummyOllama()
        )
        results["explanation"] = explanation

        # ── Step 8: Build Report ──
        progress("Building underwriting report…", 93)
        file_names = [Path(fp).name for fp in file_paths]
        report = build_report(
            classified_docs=results["classified_docs"],
            extraction_result=extraction_result,
            financial_metrics=fm,
            validation_result=val_result,
            prediction=prediction,
            explanation=explanation,
            uploaded_file_names=file_names,
        )
        results["report"] = report

        # ── Step 9: Save Outputs ──
        progress("Saving reports…", 97)
        if save_json:
            results["report_json_path"] = save_report_json(report)
        if save_pdf:
            results["report_pdf_path"] = save_report_pdf(report)

        progress("Complete!", 100)
        results["status"] = "complete"
        return results


class _DummyOllama:
    """Fallback when Ollama is unavailable."""
    def chat(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("Ollama is not available")

    def extract_json(self, prompt: str, **kwargs) -> dict:
        return {}


def _dummy_ollama():
    return _DummyOllama()

