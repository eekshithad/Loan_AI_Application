"""Quick end-to-end smoke test of the pipeline."""
import sys
sys.path.insert(0, '.')
from src.pipeline import UnderwritingPipeline
from pathlib import Path

docs = sorted(str(p) for p in Path('Input/downloaded_documents').glob('*.pdf'))
print(f'Processing {len(docs)} documents...')

pipeline = UnderwritingPipeline()

ok, msg = pipeline.check_ollama()
print(f'Ollama: {ok} - {msg}')

results = pipeline.process_files(docs, save_json=True, save_pdf=True)
print('Status:', results['status'])
if results['errors']:
    print('Errors:', results['errors'])

print()
print('=== DOCUMENT CLASSIFICATION ===')
for c in results['classified_docs']:
    print(f'  {c.file_name:45s} -> {c.category:30s} ({c.confidence:.0%}, {c.method})')

fm = results['financial_metrics']
print()
print('=== FINANCIAL METRICS ===')
print(f'  As-Is LTV : {fm.as_is_ltv}')
print(f'  ARV LTV   : {fm.arv_ltv}')
print(f'  LTC       : {fm.ltc}')
print(f'  Equity    : {fm.borrower_equity}')

pred = results['prediction']
print()
print('=== ML DECISION ===')
print(f'  Decision   : {pred["decision"]}')
print(f'  Confidence : {pred["confidence"]:.1%}')
print(f'  Probs      : {pred["probabilities"]}')

print()
print('=== VALIDATION ===')
val = results['validation_result']
print(f'  Missing required: {val.missing_required}')
print(f'  Issues count    : {len(val.issues)}')
print(f'  Completeness    : {val.overall_completeness_pct}%')

print()
print('=== REPORT FILES ===')
print(f'  JSON: {results["report_json_path"]}')
print(f'  PDF : {results["report_pdf_path"]}')
print()
print('PIPELINE SMOKE TEST DONE.')

