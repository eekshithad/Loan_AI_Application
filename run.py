"""
AI-Driven Loan Underwriting CLI Runner
Processes a loan file or folder of PDFs in a single command.

Usage:
    python run.py
    python run.py --input Input/downloaded_documents --output Output/processed_results
    python run.py --file Input/loan_application_sample.pdf
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.pipeline import UnderwritingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CLI_Runner")


def main():
    parser = argparse.ArgumentParser(description="AI-Driven Loan Underwriting CLI")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="Input/downloaded_documents",
        help="Directory containing PDF loan documents (default: Input/downloaded_documents)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a single PDF document to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="Output/processed_results",
        help="Directory to save processed results (default: Output/processed_results)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.1",
        help="Ollama LLM model name (default: llama3.1)"
    )

    args = parser.parse_args()

    # Collect PDF files
    pdf_files = []
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Specified file does not exist: {args.file}")
            sys.exit(1)
        pdf_files = [str(file_path)]
    else:
        input_dir = Path(args.input)
        if not input_dir.exists():
            # Fallback to Input
            input_dir = ROOT / "Input"
        pdf_files = sorted([str(p) for p in input_dir.glob("*.pdf")])
        if not pdf_files:
            # Check downloaded_documents subfolder
            sub_dir = input_dir / "downloaded_documents"
            if sub_dir.exists():
                pdf_files = sorted([str(p) for p in sub_dir.glob("*.pdf")])

    if not pdf_files:
        logger.error("No PDF files found to process. Please check input path.")
        sys.exit(1)

    print("=" * 70)
    print("  AI-DRIVEN LOAN UNDERWRITING SYSTEM - CLI PIPELINE")
    print(f"  Files to process: {len(pdf_files)}")
    for f in pdf_files:
        print(f"    - {Path(f).name}")
    print("=" * 70)

    # Initialize pipeline
    pipeline = UnderwritingPipeline(ollama_model=args.model)
    ollama_ok, msg = pipeline.check_ollama()
    print(f"[*] Ollama Status: {msg}")

    # Process files
    results = pipeline.process_files(
        pdf_files,
        save_json=True,
        save_pdf=True,
    )

    # Save to requested output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json_file = output_dir / "underwriting_summary.json"
    with open(result_json_file, "w", encoding="utf-8") as f:
        json.dump(results.get("report", {}), f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  UNDERWRITING ANALYSIS COMPLETE")
    print("=" * 70)

    report = results.get("report", {})
    rec = report.get("loan_recommendation", {})
    fm = report.get("financial_metrics", {})
    dc = report.get("document_classification", {})

    print(f"\n[+] Recommended Decision : {rec.get('decision', 'N/A')}")
    print(f"[+] Model Confidence     : {rec.get('confidence', 'N/A')}")
    print(f"[+] Class Probabilities  : {rec.get('probabilities', {})}")
    print(f"[+] Documents Processed  : {len(dc.get('classified', []))}")
    print(f"[+] Missing Required Docs: {dc.get('missing_required', [])}")
    print(f"\n[+] Key Financial Calculations:")
    print(f"    - As-Is LTV : {fm.get('as_is_ltv', 'N/A')}")
    print(f"    - ARV LTV   : {fm.get('arv_ltv', 'N/A')}")
    print(f"    - LTC       : {fm.get('ltc', 'N/A')}")
    print(f"    - Equity    : {fm.get('borrower_equity', 'N/A')}")

    print(f"\n[+] Saved Output Files:")
    print(f"    - JSON Result: {result_json_file}")
    if results.get("report_json_path"):
        print(f"    - Full JSON Report: {results['report_json_path']}")
    if results.get("report_pdf_path"):
        print(f"    - Formatted PDF Report: {results['report_pdf_path']}")

    print("\n[+] Human Review Items:")
    for item in report.get("human_review_requirements", []):
        print(f"    - [ ] {item}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

