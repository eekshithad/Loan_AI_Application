"""
Loan Document Generator
Generates realistic US bridge loan underwriting documents in PDF format
modeled after standard institutional templates (Fannie Mae 1003, ACORD 27, ALTA, FEMA).
"""
import logging
from pathlib import Path
from datetime import date, timedelta
import random

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("reportlab not available; cannot generate PDFs")

# Standard borrower & transaction profile
BORROWER_NAME = "Meridian Capital Investments LLC"
BORROWER_PRINCIPAL = "James A. Whitfield"
PROPERTY_ADDRESS = "1247 Oakwood Drive, Atlanta, GA 30308"
LOAN_AMOUNT = 285000
PURCHASE_PRICE = 220000
AS_IS_VALUE = 235000
ARV = 375000
REHAB_BUDGET = 95000
INTEREST_RATE = "11.5%"
LOAN_TERM = "12 months"
ORIGINATION_FEE = "2 points"
CREDIT_SCORE = 718
BANK_BALANCE = 72500
EXPERIENCE_YEARS = 5
PROJECTS_COMPLETED = 8


def _get_doc(path: str, title: str):
    """Create a SimpleDocTemplate for a document."""
    return SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        title=title,
    )


def _styles():
    return getSampleStyleSheet()


def _header_table(styles, title: str, subtitle: str = ""):
    h = ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontSize=15, textColor=colors.HexColor('#1a3c5e'),
        alignment=0, spaceAfter=2
    )
    s = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#555555'),
        spaceAfter=8
    )
    story = [Paragraph(title, h)]
    if subtitle:
        story.append(Paragraph(subtitle, s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a3c5e'), spaceAfter=10))
    return story


def _kv_table(data: list[tuple[str, str]], styles) -> Table:
    """Create a structured key-value table."""
    table_data = [[k, v] for k, v in data]
    t = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


# ── Individual Document Generators ───────────────────────────────────────────

def generate_term_sheet(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "FINACPLUS CAPITAL LLC — TERM SHEET", "Commercial Real Estate Bridge Loan Commitment")

    today = date.today()
    expiry = today + timedelta(days=14)

    kv = [
        ("Lender:", "FinacPlus Capital LLC, 100 Financial Center, Suite 1400, Atlanta, GA"),
        ("Borrower Entity:", BORROWER_NAME),
        ("Guarantor / Principal:", BORROWER_PRINCIPAL),
        ("Subject Property:", PROPERTY_ADDRESS),
        ("Loan Facility:", "First Lien Senior Secured Bridge / Renovation Loan"),
        ("Facility Amount:", f"${LOAN_AMOUNT:,}"),
        ("Interest Rate:", f"{INTEREST_RATE} per annum (Interest-Only, Monthly)"),
        ("Loan Term:", LOAN_TERM),
        ("Origination Fee:", ORIGINATION_FEE),
        ("As-Is Appraised Value:", f"${AS_IS_VALUE:,}"),
        ("After Repair Value (ARV):", f"${ARV:,}"),
        ("As-Is LTV:", f"{(LOAN_AMOUNT/AS_IS_VALUE*100):.1f}%"),
        ("ARV LTV:", f"{(LOAN_AMOUNT/ARV*100):.1f}%"),
        ("Prepayment Terms:", "3 months minimum interest guarantee; open thereafter"),
        ("Rehab Escrow / Draws:", "Rehabilitation funds held in escrow; 4 milestone disbursements subject to inspection"),
        ("Closing Date:", (today + timedelta(days=21)).strftime("%B %d, %Y")),
        ("Commitment Expiration:", expiry.strftime("%B %d, %Y")),
    ]
    story.append(_kv_table(kv, styles))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Conditions Precedent:</b> Final underwriting approval is contingent upon verification of marketable title, "
        "valid hazard/builder's risk insurance binder, satisfactory full interior appraisal report, executed operating "
        "agreement with borrowing authority, and clean background verification of the principal guarantor.",
        styles['Normal']
    ))

    doc = _get_doc(output_path, "Term Sheet")
    doc.build(story)
    return True


def generate_loan_application(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "UNIFORM COMMERCIAL LOAN APPLICATION", "Form 1003-CRE / Real Estate Investment Bridge Application")

    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10, textColor=colors.HexColor('#1a3c5e'), spaceBefore=8, spaceAfter=4)

    story.append(Paragraph("1. BORROWER ENTITY & PRINCIPAL INFORMATION", h2))
    kv1 = [
        ("Borrower Name / Entity:", BORROWER_NAME),
        ("Entity Classification:", "Domestic Limited Liability Company (LLC)"),
        ("Jurisdiction of Organization:", "Georgia (Active / Good Standing)"),
        ("Principal / Managing Member:", BORROWER_PRINCIPAL),
        ("Ownership Interest:", "85% Voting & Equity"),
        ("Primary Business Address:", "88 Peachtree Center Ave, Suite 500, Atlanta, GA 30303"),
        ("Tax ID (EIN):", "XX-XXXXXXX (On File)"),
        ("Contact Telephone & Email:", "(404) 555-0182 | jwhitfield@meridiancapinv.com"),
    ]
    story.append(_kv_table(kv1, styles))

    story.append(Paragraph("2. LOAN REQUEST & PROPERTY IDENTIFICATION", h2))
    kv2 = [
        ("Requested Loan Amount:", f"${LOAN_AMOUNT:,}"),
        ("Loan Purpose:", "Acquisition & Value-Add Rehabilitation"),
        ("Collateral Address:", PROPERTY_ADDRESS),
        ("Property Type:", "Single-Family Residential (Investment SFR)"),
        ("Contract Purchase Price:", f"${PURCHASE_PRICE:,}"),
        ("Planned Rehab Budget:", f"${REHAB_BUDGET:,}"),
        ("Projected After-Repair Value (ARV):", f"${ARV:,}"),
        ("Exit Strategy:", "Retail sale to owner-occupant upon completion of renovation"),
    ]
    story.append(_kv_table(kv2, styles))

    story.append(Paragraph("3. TRACK RECORD & FINANCIAL CAPABILITY", h2))
    kv3 = [
        ("Years in Real Estate Investment:", f"{EXPERIENCE_YEARS} years"),
        ("Completed Projects (Past 36 Months):", f"{PROJECTS_COMPLETED} completed SFR flips"),
        ("Reported Annual Entity Revenue:", "$340,000"),
        ("Current Liquid Operating Reserves:", f"${BANK_BALANCE:,}"),
    ]
    story.append(_kv_table(kv3, styles))

    doc = _get_doc(output_path, "Loan Application")
    doc.build(story)
    return True


def generate_purchase_contract(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "REAL ESTATE PURCHASE AND SALE AGREEMENT", "Standard Form — Georgia Association of REALTORS Approved")

    kv = [
        ("Purchaser / Buyer:", BORROWER_NAME),
        ("Seller:", "Estate of Robert T. Harmon (Executor: Thomas Harmon)"),
        ("Subject Property:", PROPERTY_ADDRESS),
        ("Legal Description:", "Land Lot 14, Block 7, Oakwood Subdivision, Fulton County, GA"),
        ("Purchase Price:", f"${PURCHASE_PRICE:,}"),
        ("Earnest Money Deposit:", "$5,000 held in escrow by Peachtree Title Services LLC"),
        ("Financing Contingency:", "Buyer to secure bridge/investment financing of $285,000 within 21 days"),
        ("Inspection / Due Diligence:", "14 calendar days from mutual execution"),
        ("Closing Date:", (date.today() + timedelta(days=28)).strftime("%B %d, %Y")),
        ("Closing Agent / Escrow:", "Peachtree Title Services LLC, Atlanta, GA"),
        ("Property Condition:", "As-Is, Where-Is with all faults existing at contract date"),
        ("Title Conveyance:", "Limited Warranty Deed with marketable, insurable fee-simple title"),
    ]
    story.append(_kv_table(kv, styles))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Contract Execution:</b> Duly executed by Seller and Purchaser. "
        "Earnest money receipt confirmed by escrow agent on record.",
        styles['Normal']
    ))

    doc = _get_doc(output_path, "Purchase Contract")
    doc.build(story)
    return True


def generate_appraisal(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "INDEPENDENT APPRAISAL VALUATION REPORT", "Uniform Residential Appraisal Report (Fannie Mae Form 1004 / Interagency Compliant)")

    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10, textColor=colors.HexColor('#1a3c5e'), spaceBefore=8, spaceAfter=4)

    story.append(Paragraph("1. SUBJECT PROPERTY VALUATION", h2))
    kv1 = [
        ("Subject Address:", PROPERTY_ADDRESS),
        ("County & State:", "Fulton County, GA"),
        ("Property Type & GLA:", "Single-Family Detached | 1,480 sq ft"),
        ("Room Count:", "6 Rooms, 3 Bedrooms, 2 Full Baths"),
        ("Year Built & Condition:", "1978 | Fair (Requires comprehensive interior upgrade)"),
        ("As-Is Market Value:", f"${AS_IS_VALUE:,}"),
        ("Prospective Value Upon Completion (ARV):", f"${ARV:,}"),
        ("Effective Valuation Date:", date.today().strftime("%B %d, %Y")),
        ("Certified Appraiser:", "Sandra L. Benton, MAI (GA State Cert. #GA-002847)"),
        ("Appraisal Firm:", "Benton Valuation & Advisory Group, Atlanta, GA"),
    ]
    story.append(_kv_table(kv1, styles))

    story.append(Paragraph("2. RECENT COMPARABLE CLOSED SALES", h2))
    comp_data = [
        ["Metric", "Subject", "Comp 1", "Comp 2", "Comp 3"],
        ["Address", "1247 Oakwood Dr", "1189 Maple Ave", "305 Cedar Ln", "1402 Birch St"],
        ["Sale Price", "—", "$228,000", "$241,500", "$219,000"],
        ["GLA (sq ft)", "1,480", "1,510", "1,460", "1,395"],
        ["Bed / Bath", "3 / 2", "3 / 2", "3 / 2", "3 / 1.5"],
        ["Condition", "Fair", "Good", "Good", "Average"],
        ["Adjusted Price", "—", "$232,000", "$238,000", "$235,000"],
    ]
    ct = Table(comp_data, colWidths=[1.5*inch, 1.25*inch, 1.25*inch, 1.25*inch, 1.25*inch])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
    ]))
    story.append(ct)

    doc = _get_doc(output_path, "Appraisal Report")
    doc.build(story)
    return True


def generate_scope_of_work(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "SCOPE OF WORK & CONTRACTOR REHAB BUDGET", "Detailed Line-Item Renovation Schedule")

    kv = [
        ("Project Location:", PROPERTY_ADDRESS),
        ("Owner / Sponsor:", BORROWER_NAME),
        ("General Contractor:", "Whitfield Renovation & Build LLC (GA License #GA-RC-20419)"),
        ("Projected Duration:", "90 Calendar Days (3 Construction Milestones)"),
        ("Total Renovation Budget:", f"${REHAB_BUDGET:,}"),
    ]
    story.append(_kv_table(kv, styles))
    story.append(Spacer(1, 8))

    cost_items = [
        ["Trade Category", "Detailed Specification", "Allocated Cost"],
        ["Roofing & Gutters", "Architectural shingle replacement, flashing, seamless gutters", "$14,500"],
        ["HVAC Mechanical", "New 3-ton 16-SEER heat pump, complete ductwork reconfiguration", "$9,800"],
        ["Electrical System", "200A service upgrade, panel, recessed LED lighting, GFCIs", "$7,200"],
        ["Plumbing System", "PEX re-pipe, new sewer line connection, 50-gal water heater", "$5,500"],
        ["Kitchen Renovation", "Solid shaker cabinets, quartz countertops, stainless appliance package", "$22,000"],
        ["Bathroom Remodels", "Tile surround, custom double vanity, frameless shower enclosures (2)", "$14,000"],
        ["Hardwood Flooring", "Engineered oak hardwood installation across main living areas", "$8,500"],
        ["Window Replacement", "Energy-efficient double-hung vinyl windows throughout", "$6,800"],
        ["Exterior & Landscaping", "Exterior paint, siding repair, architectural landscaping, grading", "$4,200"],
        ["Interior Paint & Trim", "Full interior paint, 5-inch baseboards, interior doors", "$3,500"],
        ["Contingency & Permits", "City of Atlanta building permits, 10% reserve contingency", "$9,000"],
        ["TOTAL BUDGET", "Turnkey Construction & Materials Total", f"${REHAB_BUDGET:,}"],
    ]
    ct = Table(cost_items, colWidths=[1.7*inch, 4.0*inch, 1.3*inch])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f0f7')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f6f8fa')]),
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
    ]))
    story.append(ct)

    doc = _get_doc(output_path, "Scope of Work")
    doc.build(story)
    return True


def generate_credit_report(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "TRI-MERGE RESIDENTIAL CREDIT REPORT", "Equifax · Experian · TransUnion Comprehensive Consumer Profile")

    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10, textColor=colors.HexColor('#1a3c5e'), spaceBefore=8, spaceAfter=4)

    story.append(Paragraph("1. BORROWER PROFILE", h2))
    kv1 = [
        ("Subject Individual:", BORROWER_PRINCIPAL),
        ("Inquiry Reference:", "CR-2024-" + str(random.randint(100000, 999999))),
        ("Report Generation Date:", date.today().strftime("%B %d, %Y")),
        ("Credit Repository Source:", "Tri-Merge Commercial Solutions LLC"),
    ]
    story.append(_kv_table(kv1, styles))

    story.append(Paragraph("2. FICO SCORES BY REPOSITORY", h2))
    score_data = [
        ["Credit Repository", "Scoring Model", "Reported Score", "Risk Tier"],
        ["Equifax", "FICO Score 8", str(CREDIT_SCORE), "Prime / Low Risk"],
        ["Experian", "FICO Score 8", str(CREDIT_SCORE - 3), "Prime / Low Risk"],
        ["TransUnion", "FICO Score 8", str(CREDIT_SCORE + 5), "Prime / Low Risk"],
        ["QUALIFYING MIDDLE SCORE", "Underwriting Decision Score", str(CREDIT_SCORE), "APPROVED TIER 1"],
    ]
    st = Table(score_data, colWidths=[2.2*inch, 1.8*inch, 1.3*inch, 1.7*inch])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f0f7')),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
    ]))
    story.append(st)

    story.append(Paragraph("3. CREDIT PROFILE SUMMARY", h2))
    kv2 = [
        ("Total Tradelines:", "7 Total (5 Active Open, 2 Closed in Good Standing)"),
        ("Public Records / Bankruptcies:", "0 Reported (Clean Public Record)"),
        ("Delinquency History (30/60/90 Days):", "0 / 0 / 0 Delinquencies"),
        ("Revolving Credit Utilization:", "19.6% ($8,240 Balance / $42,000 Available Limit)"),
        ("Length of Credit History:", "8 Years, 3 Months"),
    ]
    story.append(_kv_table(kv2, styles))

    doc = _get_doc(output_path, "Credit Report")
    doc.build(story)
    return True


def generate_bank_statement(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "PEACHTREE COMMUNITY BANK — ACCOUNT STATEMENT", "Commercial Business Checking Account")

    kv1 = [
        ("Account Title:", BORROWER_NAME),
        ("Authorized Signer:", BORROWER_PRINCIPAL),
        ("Account Number:", "XXXXXX7842"),
        ("Statement Period:", "Previous Calendar Month (30-Day Window)"),
        ("Beginning Balance:", f"${BANK_BALANCE - 12300:,}"),
        ("Total Inflows / Deposits:", "$38,500"),
        ("Total Outflows / Debits:", "$26,200"),
        ("Ending Available Balance:", f"${BANK_BALANCE:,}"),
    ]
    story.append(_kv_table(kv1, styles))
    story.append(Spacer(1, 8))

    trans_data = [
        ["Date", "Transaction Description", "Debit", "Credit", "Daily Balance"],
        ["Day 01", "Beginning Balance", "—", "—", f"${BANK_BALANCE-12300:,}"],
        ["Day 03", "Electronic Funds Transfer In", "—", "$15,000", f"${BANK_BALANCE-12300+15000:,}"],
        ["Day 07", "Contractor Material Payment", "$8,500", "—", f"${BANK_BALANCE-12300+15000-8500:,}"],
        ["Day 14", "Net Proceeds Distribution — Closed Sale", "—", "$18,000", f"${BANK_BALANCE-12300+15000-8500+18000:,}"],
        ["Day 19", "Title Escrow Deposit", "$5,000", "—", f"${BANK_BALANCE-12300+15000-8500+18000-5000:,}"],
        ["Day 22", "Commercial Insurance Premium", "$1,200", "—", f"${BANK_BALANCE-12300+15000-8500+18000-5000-1200:,}"],
        ["Day 30", "Ending Verified Cash Liquidity", "—", "—", f"${BANK_BALANCE:,}"],
    ]
    tt = Table(trans_data, colWidths=[0.8*inch, 2.7*inch, 1.0*inch, 1.0*inch, 1.5*inch])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('ALIGN', (2, 0), (4, -1), 'RIGHT'),
    ]))
    story.append(tt)

    doc = _get_doc(output_path, "Bank Statement")
    doc.build(story)
    return True


def generate_title_commitment(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "PEACHTREE TITLE SERVICES LLC — TITLE COMMITMENT", "American Land Title Association (ALTA) Plain Language Commitment")

    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10, textColor=colors.HexColor('#1a3c5e'), spaceBefore=8, spaceAfter=4)

    kv = [
        ("Commitment Number:", "PTS-2024-" + str(random.randint(10000, 99999))),
        ("Effective Date:", date.today().strftime("%B %d, %Y")),
        ("Subject Property:", PROPERTY_ADDRESS),
        ("Proposed Insured (Owner):", BORROWER_NAME),
        ("Proposed Insured (Lender):", "FinacPlus Capital LLC, its successors and/or assigns"),
        ("Loan Policy Amount:", f"${LOAN_AMOUNT:,}"),
        ("Title Status:", "Marketable Fee-Simple Title confirmed with zero delinquent tax encumbrances"),
    ]
    story.append(_kv_table(kv, styles))

    story.append(Paragraph("SCHEDULE B — SECTION 1: REQUIREMENTS", h2))
    reqs = [
        "1. Duly executed Warranty Deed from Estate of Robert T. Harmon to Meridian Capital Investments LLC.",
        "2. Recordation of first-priority Deed of Trust securing note of $285,000 to FinacPlus Capital LLC.",
        "3. Payoff and satisfaction of prior recorded lien of Sunrise Mortgage ($0 remaining balance confirmed).",
        "4. Payment of all Fulton County ad valorem property taxes due and payable through current period.",
    ]
    for r in reqs:
        story.append(Paragraph(r, styles['Normal']))

    story.append(Paragraph("SCHEDULE B — SECTION 2: EXCEPTIONS", h2))
    excs = [
        "1. Standard utility easement along northern perimeter (recorded Deed Book 418, Page 221).",
        "2. Any unrecorded mineral or water rights not reflected in public records.",
    ]
    for e in excs:
        story.append(Paragraph(e, styles['Normal']))

    doc = _get_doc(output_path, "Title Commitment")
    doc.build(story)
    return True


def generate_insurance_binder(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "EVIDENCE OF PROPERTY INSURANCE (ACORD 27)", "Commercial Dwelling & Builder's Risk Certificate")

    kv = [
        ("Insurer Carrier:", "Southern Shield Insurance Group (AM Best Rating: A Excellent)"),
        ("Policy Number:", "SSI-DP-" + str(random.randint(1000000, 9999999))),
        ("Named Insured:", BORROWER_NAME),
        ("First Mortgagee / Loss Payee:", "FinacPlus Capital LLC, ISAOA / ATIMA"),
        ("Insured Property Location:", PROPERTY_ADDRESS),
        ("Dwelling Structure Coverage:", f"${AS_IS_VALUE:,} (Guaranteed Replacement Cost)"),
        ("Builder's Risk Endorsement:", "Included — Covers all active renovation materials on site"),
        ("General Premises Liability:", "$1,000,000 per occurrence / $2,000,000 aggregate"),
        ("Policy Deductible:", "$2,500"),
        ("Policy Effective Window:", f"{date.today().strftime('%m/%d/%Y')} to {(date.today() + timedelta(days=365)).strftime('%m/%d/%Y')}"),
        ("Annual Premium Status:", "Paid in Full — 12-Month Advance Premium Secured"),
    ]
    story.append(_kv_table(kv, styles))

    doc = _get_doc(output_path, "Insurance Binder")
    doc.build(story)
    return True


def generate_flood_certificate(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "FEDERAL FLOOD HAZARD DETERMINATION (FEMA 086-0-32)", "Standard National Flood Insurance Program (NFIP) Determination")

    kv = [
        ("Subject Property Address:", PROPERTY_ADDRESS),
        ("Determination Identifier:", "NFD-" + str(random.randint(100000, 999999))),
        ("NFIP Community Name:", "City of Atlanta, Fulton County, GA (Community #130089)"),
        ("FEMA FIRM Panel Identifier:", "13089C0320F (Effective September 25, 2009)"),
        ("FEMA Flood Zone Classification:", "ZONE X (Area of Minimal Flood Hazard)"),
        ("Special Flood Hazard Area (SFHA):", "NO — Property is NOT situated in an SFHA 100-year floodplain"),
        ("Mandatory Flood Insurance Required:", "NO — Federal mandatory flood purchase requirement DOES NOT apply"),
        ("Determination Agency:", "National Flood Determination Services LLC"),
    ]
    story.append(_kv_table(kv, styles))

    doc = _get_doc(output_path, "Flood Certificate")
    doc.build(story)
    return True


def generate_operating_agreement(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "LIMITED LIABILITY COMPANY OPERATING AGREEMENT", "Articles & Regulations of Meridian Capital Investments LLC")

    kv = [
        ("Company Name:", BORROWER_NAME),
        ("State of Formation:", "Georgia (Certificate of Organization #19034821)"),
        ("Management Structure:", "Manager-Managed LLC"),
        ("Designated Sole Manager:", BORROWER_PRINCIPAL),
        ("Ownership & Equity Distribution:", f"{BORROWER_PRINCIPAL} (85%) | Diana M. Whitfield (15%)"),
        ("Manager Borrowing Authority:", "Manager has exclusive, unrestricted authority to execute promissory notes, deeds of trust, and mortgages without unanimous member resolution"),
        ("Initial Capital Contribution:", "$120,000 paid-in capital"),
        ("Business Purpose:", "Acquisition, rehabilitation, operation, and disposition of real estate assets"),
    ]
    story.append(_kv_table(kv, styles))

    doc = _get_doc(output_path, "Operating Agreement")
    doc.build(story)
    return True


def generate_borrower_track_record(output_path: str):
    if not HAS_REPORTLAB:
        return False
    styles = _styles()
    story = _header_table(styles, "SPONSOR REAL ESTATE TRACK RECORD SCHEDULE", "Verified Project History & Performance Log")

    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10, textColor=colors.HexColor('#1a3c5e'), spaceBefore=8, spaceAfter=4)

    story.append(Paragraph("1. SPONSOR SUMMARY METRICS", h2))
    kv = [
        ("Sponsor / Guarantor:", f"{BORROWER_NAME} / {BORROWER_PRINCIPAL}"),
        ("Years Active in Target Market:", f"{EXPERIENCE_YEARS} Years (Metro Atlanta, GA)"),
        ("Total Completed SFR Flips:", f"{PROJECTS_COMPLETED} Properties"),
        ("Aggregate Disposition Volume:", "$1,712,000"),
        ("Aggregate Net Realized Profit:", "$308,000"),
        ("Average Hold Period:", "4.2 Months"),
        ("Historical Default / Loan Loss:", "0 Defaults (100% On-Time Payoff History)"),
    ]
    story.append(_kv_table(kv, styles))

    story.append(Paragraph("2. RECENTLY COMPLETED REHABILITATION PROJECTS", h2))
    proj_data = [
        ["#", "Property Address", "Purchase", "Rehab", "Sale Price", "Net Profit", "Hold"],
        ["1", "214 Elm St, Atlanta GA", "$118,000", "$42,000", "$198,500", "$38,500", "5 mo"],
        ["2", "876 Pine Ave, Marietta GA", "$135,000", "$38,000", "$212,000", "$39,000", "4 mo"],
        ["3", "52 Chestnut Dr, Decatur GA", "$142,000", "$55,000", "$235,000", "$38,000", "6 mo"],
        ["4", "1801 Oak St, Savannah GA", "$98,000", "$35,000", "$169,000", "$36,000", "4 mo"],
        ["5", "420 Maple Rd, Macon GA", "$112,000", "$48,000", "$198,000", "$38,000", "5 mo"],
        ["6", "703 Birch Ln, Albany GA", "$88,000", "$32,000", "$152,000", "$32,000", "3 mo"],
        ["7", "55 Willow Way, Tampa FL", "$155,000", "$62,000", "$258,000", "$41,000", "5 mo"],
        ["8", "19 Cypress Ct, Nashville TN", "$128,000", "$44,000", "$218,000", "$46,000", "4 mo"],
    ]
    pt = Table(proj_data, colWidths=[0.3*inch, 2.3*inch, 0.8*inch, 0.75*inch, 0.95*inch, 0.85*inch, 0.55*inch])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
    ]))
    story.append(pt)

    doc = _get_doc(output_path, "Borrower Track Record")
    doc.build(story)
    return True


DOCUMENT_GENERATORS = {
    "term_sheet_sample.pdf": generate_term_sheet,
    "loan_application_sample.pdf": generate_loan_application,
    "purchase_contract_sample.pdf": generate_purchase_contract,
    "appraisal_sample.pdf": generate_appraisal,
    "scope_of_work_sample.pdf": generate_scope_of_work,
    "credit_report_sample.pdf": generate_credit_report,
    "bank_statement_sample.pdf": generate_bank_statement,
    "title_commitment_sample.pdf": generate_title_commitment,
    "insurance_binder_sample.pdf": generate_insurance_binder,
    "flood_certificate_sample.pdf": generate_flood_certificate,
    "operating_agreement_sample.pdf": generate_operating_agreement,
    "borrower_track_record_sample.pdf": generate_borrower_track_record,
}

DOCUMENT_SOURCES = [
    {
        "file_name": "term_sheet_sample.pdf",
        "category": "Term Sheet",
        "source_type": "Institutional Standard",
        "url": "Commercial Bridge Lending Standard Template",
        "source_name": "FinacPlus Capital Commercial Credit Division",
        "rationale": "Specifies fundamental loan terms: loan amount, note rate, maturity, and initial LTV thresholds.",
        "limitations": "Standard 12-month bridge format without subordinate financing clauses.",
    },
    {
        "file_name": "loan_application_sample.pdf",
        "category": "Loan Application",
        "source_type": "Standard Form",
        "url": "Fannie Mae Form 1003 / Uniform Commercial Loan Application Reference",
        "source_name": "Mortgage Bankers Association (MBA) Commercial Format",
        "rationale": "Provides primary borrower identity, entity organization, requested capital, and experience details.",
        "limitations": "Guarantor personal PII (SSN/DOB) masked for privacy protection.",
    },
    {
        "file_name": "purchase_contract_sample.pdf",
        "category": "Purchase Contract",
        "source_type": "Standard Real Estate Contract",
        "url": "Georgia Association of REALTORS (GAR) Standard Agreement",
        "source_name": "Peachtree Escrow & Settlement Services",
        "rationale": "Legal baseline establishing contract purchase price, earnest money deposit, and closing timelines.",
        "limitations": "Standard 'As-Is' residential purchase form without municipal escrow holdbacks.",
    },
    {
        "file_name": "appraisal_sample.pdf",
        "category": "Appraisal",
        "source_type": "Certified Valuation Report",
        "url": "Uniform Residential Appraisal Report (URAR Form 1004)",
        "source_name": "Benton Valuation & Advisory Group (MAI Certified)",
        "rationale": "Independent expert valuation providing both As-Is property value and post-renovation ARV.",
        "limitations": "Uses 3 direct neighborhood comps; full market trend addendum condensed.",
    },
    {
        "file_name": "scope_of_work_sample.pdf",
        "category": "Scope of Work",
        "source_type": "Contractor Schedule",
        "url": "Standard Residential Construction Trade Schedule",
        "source_name": "Whitfield Renovation & Build LLC (Licensed General Contractor)",
        "rationale": "Itemized line-item construction budget necessary to establish LTC and rehab viability.",
        "limitations": "Fixed-bid contractor schedule subject to 10% standard contingency reserve.",
    },
    {
        "file_name": "credit_report_sample.pdf",
        "category": "Credit Report",
        "source_type": "Tri-Merge Bureau Record",
        "url": "Equifax, Experian, TransUnion Standard Tri-Merge Repository",
        "source_name": "Tri-Merge Commercial Solutions",
        "rationale": "Determines guarantor creditworthiness, middle FICO score, and credit utilization history.",
        "limitations": "Condensed summary view of tradelines without individual historical billing statements.",
    },
    {
        "file_name": "bank_statement_sample.pdf",
        "category": "Bank Statement",
        "source_type": "Verified Depository Record",
        "url": "Peachtree Community Bank Commercial Depository Ledger",
        "source_name": "Peachtree Community Bank Depository Services",
        "rationale": "Verifies available cash reserves and liquidity to satisfy equity and interest requirements.",
        "limitations": "Single 30-day billing statement cycle.",
    },
    {
        "file_name": "title_commitment_sample.pdf",
        "category": "Title Commitment",
        "source_type": "ALTA Plain Language Format",
        "url": "American Land Title Association Standard Commitment Form",
        "source_name": "Peachtree Title Services LLC",
        "rationale": "Identifies legal ownership, encumbrances, liens, and title insurability requirements.",
        "limitations": "Schedule B requirements assume standard deed of trust payoff at closing.",
    },
    {
        "file_name": "insurance_binder_sample.pdf",
        "category": "Insurance Binder",
        "source_type": "ACORD 27 Standard",
        "url": "ACORD Form 27 — Evidence of Property Insurance",
        "source_name": "Southern Shield Insurance Group",
        "rationale": "Confirms active property hazard and builder's risk insurance naming lender as loss payee.",
        "limitations": "Annual policy binder; final premium receipt acknowledged.",
    },
    {
        "file_name": "flood_certificate_sample.pdf",
        "category": "Flood Certificate",
        "source_type": "FEMA Form 086-0-32",
        "url": "Standard Flood Hazard Determination Form",
        "source_name": "National Flood Determination Services LLC",
        "rationale": "Verifies whether collateral is in Special Flood Hazard Area (SFHA Zone X confirmed).",
        "limitations": "FIRM panel map date September 2009.",
    },
    {
        "file_name": "operating_agreement_sample.pdf",
        "category": "Operating Agreement",
        "source_type": "Legal Corporate Document",
        "url": "Georgia Secretary of State LLC Organizational Template",
        "source_name": "Corporate Counsel / Legal Records",
        "rationale": "Establishes entity structure, equity ownership, and manager authority to execute debt.",
        "limitations": "Standard manager-managed two-member structure.",
    },
    {
        "file_name": "borrower_track_record_sample.pdf",
        "category": "Borrower Track Record",
        "source_type": "Verified Portfolio Schedule",
        "url": "Institutional Investor Historical Project Log",
        "source_name": "Meridian Capital Investments Portfolio Archive",
        "rationale": "Documents 8 completed fix-and-flip projects confirming investor execution capability.",
        "limitations": "Reflects past 36 months closed transactions.",
    },
]


def generate_all_documents(output_dir: str = "Input/downloaded_documents") -> list[str]:
    """Generate clean, authentic loan documents in the specified directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created = []
    for filename, generator_fn in DOCUMENT_GENERATORS.items():
        path = out / filename
        try:
            success = generator_fn(str(path))
            if success:
                logger.info(f"Generated clean document: {filename}")
                created.append(str(path))
            else:
                logger.warning(f"Failed to generate: {filename}")
        except Exception as exc:
            logger.error(f"Error generating {filename}: {exc}")

    return created


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generated = generate_all_documents()
    print(f"Generated {len(generated)} clean loan documents.")
