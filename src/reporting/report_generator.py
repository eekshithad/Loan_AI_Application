"""
Report Generation Module
Generates structured underwriting reports as JSON and PDF.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("Output/reports")


def _safe_val(field_obj) -> str:
    """Safely get value from ExtractedField."""
    if field_obj is None:
        return "Not Found"
    val = getattr(field_obj, 'value', "Not Found")
    return str(val) if val is not None else "Not Found"


def _safe_src(field_obj) -> str:
    """Safely get source from ExtractedField."""
    if field_obj is None:
        return "N/A"
    doc = getattr(field_obj, 'source_document', "N/A")
    page = getattr(field_obj, 'source_page', 0)
    if doc and doc != "N/A" and page:
        return f"{doc} (p.{page})"
    return doc or "N/A"


def build_report(
    classified_docs: list,
    extraction_result,
    financial_metrics,
    validation_result,
    prediction: dict,
    explanation: str,
    uploaded_file_names: list[str],
) -> dict:
    """
    Build a complete underwriting report as a Python dict.
    """
    ext = extraction_result
    fm = financial_metrics
    val = validation_result

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "metadata": {
            "generated_at": now,
            "application_name": "AI-Driven Loan Underwriting System",
            "version": "1.0.0",
            "disclaimer": "Confidential Commercial Underwriting Report — FinacPlus Credit Division",
        },
        "loan_summary": {
            "property_address": _safe_val(ext.loan.property_address),
            "borrower_name": _safe_val(ext.borrower.name),
            "loan_amount": _safe_val(ext.loan.loan_amount),
            "purchase_price": _safe_val(ext.loan.purchase_price),
            "as_is_value": _safe_val(ext.loan.as_is_value),
            "arv": _safe_val(ext.loan.arv),
            "rehab_budget": _safe_val(ext.loan.rehab_budget),
        },
        "document_classification": {
            "uploaded_files": uploaded_file_names,
            "classified": [
                {
                    "file": c.file_name,
                    "tag": getattr(c, "tag", "other"),
                    "category": c.category,
                    "confidence": f"{c.confidence:.0%}",
                    "method": c.method,
                }
                for c in classified_docs
            ],
            "missing_required": val.missing_required,
            "missing_recommended": val.missing_recommended,
            "completeness_pct": val.overall_completeness_pct,
        },
        "extracted_information": {
            "borrower": {
                "name": {"value": _safe_val(ext.borrower.name), "source": _safe_src(ext.borrower.name)},
                "entity_type": {"value": _safe_val(ext.borrower.entity_type), "source": _safe_src(ext.borrower.entity_type)},
                "credit_score": {"value": _safe_val(ext.borrower.credit_score), "source": _safe_src(ext.borrower.credit_score)},
                "employment": {"value": _safe_val(ext.borrower.employment), "source": _safe_src(ext.borrower.employment)},
                "bank_balance": {"value": _safe_val(ext.borrower.bank_balance), "source": _safe_src(ext.borrower.bank_balance)},
                "experience_years": {"value": _safe_val(ext.borrower.experience_years), "source": _safe_src(ext.borrower.experience_years)},
                "projects_completed": {"value": _safe_val(ext.borrower.projects_completed), "source": _safe_src(ext.borrower.projects_completed)},
            },
            "loan": {
                "loan_amount": {"value": _safe_val(ext.loan.loan_amount), "source": _safe_src(ext.loan.loan_amount)},
                "purchase_price": {"value": _safe_val(ext.loan.purchase_price), "source": _safe_src(ext.loan.purchase_price)},
                "as_is_value": {"value": _safe_val(ext.loan.as_is_value), "source": _safe_src(ext.loan.as_is_value)},
                "arv": {"value": _safe_val(ext.loan.arv), "source": _safe_src(ext.loan.arv)},
                "rehab_budget": {"value": _safe_val(ext.loan.rehab_budget), "source": _safe_src(ext.loan.rehab_budget)},
                "property_address": {"value": _safe_val(ext.loan.property_address), "source": _safe_src(ext.loan.property_address)},
                "interest_rate": {"value": _safe_val(ext.loan.interest_rate), "source": _safe_src(ext.loan.interest_rate)},
                "loan_term_months": {"value": _safe_val(ext.loan.loan_term_months), "source": _safe_src(ext.loan.loan_term_months)},
                "origination_fee": {"value": _safe_val(ext.loan.origination_fee), "source": _safe_src(ext.loan.origination_fee)},
            },
            "supporting": {
                "insurance_carrier": {"value": _safe_val(ext.supporting.insurance_carrier), "source": _safe_src(ext.supporting.insurance_carrier)},
                "insurance_amount": {"value": _safe_val(ext.supporting.insurance_amount), "source": _safe_src(ext.supporting.insurance_amount)},
                "flood_zone": {"value": _safe_val(ext.supporting.flood_zone), "source": _safe_src(ext.supporting.flood_zone)},
                "title_issues": {"value": _safe_val(ext.supporting.title_issues), "source": _safe_src(ext.supporting.title_issues)},
                "property_type": {"value": _safe_val(ext.supporting.property_type), "source": _safe_src(ext.supporting.property_type)},
            },
        },
        "financial_metrics": {
            "as_is_ltv": f"{fm.as_is_ltv:.2f}%" if fm.as_is_ltv else "N/A",
            "arv_ltv": f"{fm.arv_ltv:.2f}%" if fm.arv_ltv else "N/A",
            "ltc": f"{fm.ltc:.2f}%" if fm.ltc else "N/A",
            "total_project_cost": f"${fm.total_project_cost:,.0f}" if fm.total_project_cost else "N/A",
            "borrower_equity": f"${fm.borrower_equity:,.0f}" if fm.borrower_equity else "N/A",
            "risk_flags": fm.flags,
        },
        "validation_issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "description": i.description,
                "action": i.suggested_action,
            }
            for i in val.issues
        ],
        "loan_recommendation": {
            "decision": prediction.get("decision", "N/A"),
            "confidence": f"{prediction.get('confidence', 0):.1%}",
            "probabilities": prediction.get("probabilities", {}),
            "top_factors": prediction.get("top_features", []),
            "explanation": explanation,
            "model_disclosure": "Quantitative risk score derived from multi-variable Random Forest classifier.",
        },
        "human_review_requirements": _build_human_review(val, prediction),
    }

    return report


def _build_human_review(val, prediction: dict) -> list[str]:
    """List items requiring human review."""
    items = []

    if val.missing_required:
        items.append(f"Obtain missing required documents: {', '.join(val.missing_required)}")

    errors = [i for i in val.issues if i.severity == "error"]
    for err in errors[:3]:
        items.append(f"Resolve: {err.description}")

    if prediction.get("decision") == "CONDITIONAL_APPROVAL":
        items.append("Review and document all conditions before funding")
        items.append("Obtain borrower acknowledgment of conditions")

    if prediction.get("decision") == "DECLINE":
        items.append("Provide adverse action notice to borrower")
        items.append("Document specific decline reasons per regulatory requirements")

    if not items:
        items.append("Standard final review by senior underwriter before funding")

    return items


def save_report_json(report: dict, output_dir: str = "Output/reports") -> str:
    """Save report as JSON file. Returns the file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"underwriting_report_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {path}")
    return str(path)


def save_report_pdf(report: dict, output_dir: str = "Output/reports") -> str:
    """Save report as PDF using reportlab. Returns the file path."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"underwriting_report_{timestamp}.pdf"

        doc = SimpleDocTemplate(str(path), pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'Title', parent=styles['Title'],
            fontSize=18, spaceAfter=6, textColor=colors.HexColor('#1a3c5e'),
        )
        h1_style = ParagraphStyle(
            'H1', parent=styles['Heading1'],
            fontSize=13, textColor=colors.HexColor('#1a3c5e'),
            spaceBefore=12, spaceAfter=4,
        )
        h2_style = ParagraphStyle(
            'H2', parent=styles['Heading2'],
            fontSize=11, textColor=colors.HexColor('#2c5f8a'),
            spaceBefore=8, spaceAfter=3,
        )
        body_style = styles['Normal']
        body_style.fontSize = 9
        small_style = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.gray)

        def hr():
            return HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6)

        def table(data, col_widths=None, header_row=True):
            t = Table(data, colWidths=col_widths)
            ts = [
                ('BACKGROUND', (0, 0), (-1, 0 if header_row else -1),
                 colors.HexColor('#e8f0f7')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a1a')),
                ('FONTNAME', (0, 0), (-1, 0 if header_row else -1), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1 if header_row else 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9fc')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]
            t.setStyle(TableStyle(ts))
            return t

        # ── Header ──
        story.append(Paragraph("AI-Driven Loan Underwriting Report", title_style))
        story.append(Paragraph(
            f"Generated: {report['metadata']['generated_at']}", small_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(report['metadata']['disclaimer'], small_style))
        story.append(hr())

        # ── 1. Loan Summary ──
        story.append(Paragraph("1. Loan Application Summary", h1_style))
        ls = report['loan_summary']
        summary_data = [
            ["Field", "Value"],
            ["Property Address", ls['property_address']],
            ["Borrower Name", ls['borrower_name']],
            ["Loan Amount", ls['loan_amount']],
            ["Purchase Price", ls['purchase_price']],
            ["As-Is Property Value", ls['as_is_value']],
            ["After Repair Value (ARV)", ls['arv']],
            ["Rehabilitation Budget", ls['rehab_budget']],
        ]
        story.append(table(summary_data, col_widths=[2.5*inch, 4.5*inch]))
        story.append(Spacer(1, 8))

        # ── 2. Document Classification ──
        story.append(Paragraph("2. Document Classification", h1_style))
        dc = report['document_classification']
        story.append(Paragraph(
            f"Completeness: {dc['completeness_pct']}% | "
            f"Documents uploaded: {len(dc['uploaded_files'])}", body_style))
        story.append(Spacer(1, 4))

        cls_data = [["Document", "Category", "Confidence", "Method"]]
        for c in dc['classified']:
            cls_data.append([c['file'][:40], c['category'], c['confidence'], c['method']])
        story.append(table(cls_data, col_widths=[2.5*inch, 2*inch, 1*inch, 1.5*inch]))

        if dc['missing_required']:
            story.append(Spacer(1, 4))
            story.append(Paragraph("⚠️ Missing Required Documents:", h2_style))
            for m in dc['missing_required']:
                story.append(Paragraph(f"• {m}", body_style))

        story.append(Spacer(1, 8))

        # ── 3. Financial Metrics ──
        story.append(Paragraph("3. Financial Calculations", h1_style))
        fm_data = report['financial_metrics']
        metrics_table = [
            ["Metric", "Value", "Threshold"],
            ["As-Is LTV", fm_data['as_is_ltv'], "≤ 80%"],
            ["ARV LTV", fm_data['arv_ltv'], "≤ 75%"],
            ["Loan-to-Cost (LTC)", fm_data['ltc'], "≤ 90%"],
            ["Total Project Cost", fm_data['total_project_cost'], "—"],
            ["Borrower Equity", fm_data['borrower_equity'], "> $0"],
        ]
        story.append(table(metrics_table, col_widths=[2.5*inch, 2*inch, 2.5*inch]))

        if fm_data['risk_flags']:
            story.append(Spacer(1, 4))
            story.append(Paragraph("Risk Flags:", h2_style))
            for flag in fm_data['risk_flags']:
                story.append(Paragraph(f"• {flag}", body_style))

        story.append(Spacer(1, 8))

        # ── 4. Validation Issues ──
        story.append(Paragraph("4. Validation & Consistency Issues", h1_style))
        vi = report['validation_issues']
        if vi:
            issues_data = [["Severity", "Code", "Description"]]
            for issue in vi:
                issues_data.append([
                    issue['severity'].upper(),
                    issue['code'],
                    issue['description'][:60],
                ])
            story.append(table(issues_data, col_widths=[1*inch, 2*inch, 4*inch]))
        else:
            story.append(Paragraph("No significant issues found.", body_style))

        story.append(Spacer(1, 8))
        story.append(PageBreak())

        # ── 5. Loan Recommendation ──
        story.append(Paragraph("5. AI/ML Loan Recommendation", h1_style))
        rec = report['loan_recommendation']

        decision_color = {
            "APPROVE": colors.HexColor('#2ecc71'),
            "CONDITIONAL_APPROVAL": colors.HexColor('#f39c12'),
            "DECLINE": colors.HexColor('#e74c3c'),
        }.get(rec['decision'], colors.HexColor('#333333'))

        decision_style = ParagraphStyle(
            'Decision', parent=styles['Normal'],
            fontSize=16, fontName='Helvetica-Bold',
            textColor=decision_color, spaceAfter=6,
        )
        story.append(Paragraph(f"Decision: {rec['decision']}", decision_style))
        story.append(Paragraph(f"Model Confidence: {rec['confidence']}", body_style))
        story.append(Spacer(1, 6))

        # Probabilities
        probs = rec['probabilities']
        prob_data = [["Decision Category", "Probability"]]
        for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            prob_data.append([label, f"{prob:.1%}"])
        story.append(table(prob_data, col_widths=[3*inch, 2*inch]))
        story.append(Spacer(1, 6))

        # Top factors
        if rec['top_factors']:
            story.append(Paragraph("Key Factors Influencing Decision:", h2_style))
            factors_data = [["Factor", "Importance"]]
            for f in rec['top_factors']:
                factors_data.append([f['feature'].replace('_', ' ').title(), f"{f['importance']:.1%}"])
            story.append(table(factors_data, col_widths=[3*inch, 2*inch]))
        story.append(Spacer(1, 8))

        # Explanation
        story.append(Paragraph("Underwriter Explanation:", h2_style))
        story.append(Paragraph(rec['explanation'].replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph(rec['model_disclosure'], small_style))
        story.append(hr())

        # ── 6. Human Review ──
        story.append(Paragraph("6. Human Review Requirements", h1_style))
        for item in report['human_review_requirements']:
            story.append(Paragraph(f"☐ {item}", body_style))

        doc.build(story)
        logger.info(f"PDF report saved: {path}")
        return str(path)

    except Exception as exc:
        logger.error(f"PDF generation failed: {exc}")
        # Fall back to JSON
        return save_report_json(report, output_dir)

