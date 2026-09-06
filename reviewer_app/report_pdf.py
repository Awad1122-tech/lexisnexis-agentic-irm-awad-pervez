"""
report_pdf.py - Professional PDF Report Generator

Generates a clean, professional PDF report for the Senior Analyst
from the FinalReport object. The PDF is the complete, self-contained
document the Senior Analyst receives to make their final decision.

Uses ReportLab's Platypus framework for structured, multi-page documents.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from schemas import FinalReport

# -----------------------------------------------------------------------
# Colour palette — LexisNexis brand colours
# -----------------------------------------------------------------------
LN_ORANGE = colors.HexColor('#E8500A')       # LexisNexis primary orange
LN_DARK = colors.HexColor('#1A1A2E')         # Dark navy for headings
LN_LIGHT_GREY = colors.HexColor('#F0F2F5')   # Light grey for backgrounds
LN_MID_GREY = colors.HexColor('#888888')     # Mid grey for labels
LN_GREEN = colors.HexColor('#155724')        # Compliant green
LN_RED = colors.HexColor('#721C24')          # Non-compliant red
LN_AMBER = colors.HexColor('#856404')        # Pending amber
LN_WHITE = colors.white

# -----------------------------------------------------------------------
# Risk rating colours
# -----------------------------------------------------------------------
RISK_COLOURS = {
    'LOW': colors.HexColor('#155724'),
    'MEDIUM': colors.HexColor('#856404'),
    'MEDIUM-HIGH': colors.HexColor('#7D3C00'),
    'HIGH': colors.HexColor('#721C24'),
}

RISK_BG_COLOURS = {
    'LOW': colors.HexColor('#D4EDDA'),
    'MEDIUM': colors.HexColor('#FFF3CD'),
    'MEDIUM-HIGH': colors.HexColor('#FFE5D0'),
    'HIGH': colors.HexColor('#F8D7DA'),
}


def build_styles():
    """
    Builds and returns all custom paragraph styles used in the report.
    Based on ReportLab's default stylesheet, extended with custom styles.
    """
    styles = getSampleStyleSheet()

    # Cover page title
    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=LN_DARK,
        alignment=TA_CENTER,
        spaceAfter=6
    ))

    # Cover page subtitle
    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        fontName='Helvetica',
        fontSize=13,
        textColor=LN_MID_GREY,
        alignment=TA_CENTER,
        spaceAfter=4
    ))

    # Cover page branding
    styles.add(ParagraphStyle(
        name='CoverBrand',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=LN_ORANGE,
        alignment=TA_CENTER,
        spaceAfter=2,
        letterSpacing=1
    ))

    # Section heading
    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=LN_ORANGE,
        spaceBefore=14,
        spaceAfter=6,
        letterSpacing=0.5
    ))

    # Field label
    styles.add(ParagraphStyle(
        name='FieldLabel',
        fontName='Helvetica-Bold',
        fontSize=8,
        textColor=LN_MID_GREY,
        spaceAfter=1
    ))

    # Field value
    styles.add(ParagraphStyle(
        name='FieldValue',
        fontName='Helvetica',
        fontSize=10,
        textColor=LN_DARK,
        spaceAfter=6
    ))

    # Body text
    styles.add(ParagraphStyle(
        name='BodyText2',
        fontName='Helvetica',
        fontSize=9,
        textColor=LN_DARK,
        spaceAfter=6,
        leading=14
    ))

    # Bullet item
    styles.add(ParagraphStyle(
        name='BulletItem',
        fontName='Helvetica',
        fontSize=9,
        textColor=LN_DARK,
        spaceAfter=3,
        leftIndent=12,
        leading=13
    ))

    # Footer text
    styles.add(ParagraphStyle(
        name='FooterText',
        fontName='Helvetica',
        fontSize=7,
        textColor=LN_MID_GREY,
        alignment=TA_CENTER
    ))

    # Compliance status
    styles.add(ParagraphStyle(
        name='ComplianceTrue',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=LN_GREEN,
        spaceAfter=3
    ))

    styles.add(ParagraphStyle(
        name='ComplianceFalse',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=LN_RED,
        spaceAfter=3
    ))

    styles.add(ParagraphStyle(
        name='CompliancePending',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=LN_AMBER,
        spaceAfter=3
    ))

    return styles


def section_divider():
    """Returns a styled horizontal rule for section separation."""
    return HRFlowable(
        width="100%",
        thickness=1,
        color=LN_LIGHT_GREY,
        spaceAfter=8,
        spaceBefore=4
    )


def orange_divider():
    """Returns a thin orange rule for section headings."""
    return HRFlowable(
        width="100%",
        thickness=2,
        color=LN_ORANGE,
        spaceAfter=8,
        spaceBefore=2
    )


def compliance_status_text(value, styles):
    """
    Returns a colour-coded compliance status paragraph.
    Green for compliant, red for non-compliant, amber for pending.
    """
    if value is True:
        return Paragraph("COMPLIANT", styles['ComplianceTrue'])
    elif value is False:
        return Paragraph("NON-COMPLIANT", styles['ComplianceFalse'])
    elif value == 'Compliant':
        return Paragraph("COMPLIANT", styles['ComplianceTrue'])
    elif value == 'Non-Compliant':
        return Paragraph("NON-COMPLIANT", styles['ComplianceFalse'])
    elif value == 'Pending Verification':
        return Paragraph("PENDING VERIFICATION", styles['CompliancePending'])
    else:
        return Paragraph(str(value), styles['FieldValue'])


def build_cover_page(report: FinalReport, styles) -> list:
    """
    Builds the cover page elements.
    Returns a list of ReportLab flowables.
    """
    elements = []

    # Top spacing
    elements.append(Spacer(1, 40*mm))

    # LexisNexis branding
    elements.append(Paragraph(
        "LEXISNEXIS RISK SOLUTIONS",
        styles['CoverBrand']
    ))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "Agentic Identity Risk Management System",
        styles['CoverSubtitle']
    ))

    # Orange divider
    elements.append(Spacer(1, 8*mm))
    elements.append(HRFlowable(
        width="60%",
        thickness=3,
        color=LN_ORANGE,
        hAlign='CENTER',
        spaceAfter=8*mm
    ))

    # Report title
    elements.append(Paragraph(
        "IDENTITY RISK ASSESSMENT REPORT",
        styles['CoverTitle']
    ))
    elements.append(Spacer(1, 4*mm))

    # Case ID and customer name
    elements.append(Paragraph(
        f"{report.case_id}",
        ParagraphStyle(
            'CaseID',
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=LN_ORANGE,
            alignment=TA_CENTER,
            spaceAfter=4
        )
    ))
    elements.append(Paragraph(
        report.customer_name,
        ParagraphStyle(
            'CustomerName',
            fontName='Helvetica',
            fontSize=14,
            textColor=LN_DARK,
            alignment=TA_CENTER,
            spaceAfter=8
        )
    ))

    # Risk rating badge (table for background colour)
    rating = report.overall_risk_rating
    risk_bg = RISK_BG_COLOURS.get(rating, LN_LIGHT_GREY)
    risk_fg = RISK_COLOURS.get(rating, LN_DARK)

    risk_table = Table(
        [[Paragraph(
            f"OVERALL RISK: {rating}",
            ParagraphStyle(
                'RiskBadge',
                fontName='Helvetica-Bold',
                fontSize=14,
                textColor=risk_fg,
                alignment=TA_CENTER
            )
        )]],
        colWidths=[80*mm]
    )
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), risk_bg),
        ('ROUNDEDCORNERS', [6]),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    elements.append(Spacer(1, 6*mm))
    elements.append(Table(
        [[risk_table]],
        colWidths=[160*mm],
        style=TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')])
    ))

    # Report metadata
    elements.append(Spacer(1, 12*mm))

    meta_data = [
        ['Report Generated:', report.report_timestamp[:19].replace('T', ' ')],
        ['Audit Trail Reference:', report.audit_trail_reference or 'N/A'],
        ['Prepared By:', 'LexisNexis Agentic Identity Risk Management System'],
        ['Classification:', 'CONFIDENTIAL — FOR SENIOR ANALYST USE ONLY'],
    ]

    meta_table = Table(meta_data, colWidths=[50*mm, 110*mm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), LN_MID_GREY),
        ('TEXTCOLOR', (1, 0), (1, -1), LN_DARK),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ]))
    elements.append(meta_table)

    # Decision box for Senior Analyst
    elements.append(Spacer(1, 12*mm))

    decision_box = Table(
        [[Paragraph(
            "SENIOR ANALYST DECISION",
            ParagraphStyle(
                'DecisionHeader',
                fontName='Helvetica-Bold',
                fontSize=9,
                textColor=LN_MID_GREY,
                alignment=TA_CENTER
            )
        )],
        [Paragraph(
            "  [ ]  APPROVE          [ ]  REJECT          [ ]  ESCALATE  ",
            ParagraphStyle(
                'DecisionOptions',
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=LN_DARK,
                alignment=TA_CENTER
            )
        )],
        [Paragraph(
            "Signature: ___________________________     Date: ________________",
            ParagraphStyle(
                'DecisionSignature',
                fontName='Helvetica',
                fontSize=9,
                textColor=LN_DARK,
                alignment=TA_CENTER
            )
        )]],
        colWidths=[160*mm]
    )
    decision_box.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.5, LN_ORANGE),
        ('BACKGROUND', (0, 0), (-1, 0), LN_LIGHT_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('LINEBELOW', (0, 0), (-1, 0), 1, LN_ORANGE),
        ('LINEBELOW', (0, 1), (-1, 1), 0.5, LN_LIGHT_GREY),
    ]))
    elements.append(decision_box)

    elements.append(PageBreak())
    return elements


def build_executive_summary(report: FinalReport, styles) -> list:
    """Builds the executive summary and decision rationale section."""
    elements = []

    # Section heading
    elements.append(Paragraph("1. EXECUTIVE SUMMARY & DECISION", styles['SectionHeading']))
    elements.append(orange_divider())

    # Executive summary
    elements.append(Paragraph("Executive Summary", styles['FieldLabel']))
    elements.append(Paragraph(report.executive_summary, styles['BodyText2']))
    elements.append(Spacer(1, 4*mm))

    # Decision rationale
    elements.append(Paragraph("Decision Rationale", styles['FieldLabel']))
    elements.append(Paragraph(report.decision_rationale, styles['BodyText2']))
    elements.append(Spacer(1, 4*mm))

    # Final recommendations
    elements.append(Paragraph("Final Recommendations", styles['FieldLabel']))
    for rec in report.final_recommendations:
        elements.append(Paragraph(f"• {rec}", styles['BulletItem']))
    elements.append(Spacer(1, 4*mm))

    # Final next steps
    elements.append(Paragraph("Final Next Steps", styles['FieldLabel']))
    for step in report.final_next_steps:
        elements.append(Paragraph(f"• {step}", styles['BulletItem']))

    elements.append(section_divider())
    return elements


def build_compliance_section(report: FinalReport, styles) -> list:
    """Builds the compliance status section."""
    elements = []

    elements.append(Paragraph("2. COMPLIANCE STATUS", styles['SectionHeading']))
    elements.append(orange_divider())

    cs = report.compliance_status

    # Compliance overview table
    compliance_data = [
        [
            Paragraph("GDPR", styles['FieldLabel']),
            Paragraph("FCA", styles['FieldLabel']),
            Paragraph("AML/KYC", styles['FieldLabel'])
        ],
        [
            compliance_status_text(cs.gdpr_compliant, styles),
            compliance_status_text(cs.fca_compliant, styles),
            compliance_status_text(cs.aml_kyc_compliant, styles)
        ]
    ]

    compliance_table = Table(compliance_data, colWidths=[53*mm, 53*mm, 54*mm])
    compliance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LN_LIGHT_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 0.5, LN_MID_GREY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LN_LIGHT_GREY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(compliance_table)
    elements.append(Spacer(1, 4*mm))

    # Compliance flags
    elements.append(Paragraph("Compliance Flags", styles['FieldLabel']))
    for flag in cs.compliance_flags:
        elements.append(Paragraph(f"• {flag}", styles['BulletItem']))
    elements.append(Spacer(1, 4*mm))

    # Regulatory actions required
    elements.append(Paragraph("Regulatory Actions Required", styles['FieldLabel']))
    for action in cs.regulatory_actions_required:
        elements.append(Paragraph(f"• {action}", styles['BulletItem']))

    # Mortality block
    if cs.mortality_confirmed_block:
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph(
            "MORTALITY CONFIRMED BLOCK: This case has been blocked due to a confirmed mortality match.",
            ParagraphStyle(
                'MortalityBlock',
                fontName='Helvetica-Bold',
                fontSize=9,
                textColor=LN_RED,
                spaceAfter=4
            )
        ))

    elements.append(section_divider())
    return elements


def build_risk_assessment_section(report: FinalReport, styles) -> list:
    """Builds the detailed risk assessment section."""
    elements = []

    elements.append(Paragraph("3. RISK ASSESSMENT DETAILS", styles['SectionHeading']))
    elements.append(orange_divider())

    ra = report.risk_assessment

    # Overview table
    overview_data = [
        ['Risk Score', 'Trust Score', 'Confidence Score', 'Confidence Level'],
        [
            str(ra.risk_score),
            str(ra.trust_score) if ra.trust_score is not None else 'N/A',
            str(ra.confidence_score),
            ra.confidence_level
        ]
    ]

    overview_table = Table(overview_data, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LN_LIGHT_GREY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, 0), LN_MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, LN_MID_GREY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LN_LIGHT_GREY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 4*mm))

    # Key signals
    elements.append(Paragraph("Key Risk Signals", styles['FieldLabel']))
    for signal in ra.key_signals:
        elements.append(Paragraph(f"• {signal}", styles['BulletItem']))
    elements.append(Spacer(1, 4*mm))

    # Identity & document verification
    elements.append(Paragraph("Identity & Document Verification", styles['FieldLabel']))

    id_data = [
        ['Identity Verification', 'Document Type', 'Document Authentication', 'Auth Score'],
        [
            ra.identity_verification_result,
            ra.document_type or 'N/A',
            ra.document_authentication_result,
            str(ra.document_authentication_score)
        ],
        ['Facial Match Score', 'Liveness Check', '', ''],
        [
            str(ra.facial_match_score),
            'Passed' if ra.liveness_check_passed else 'Failed',
            '', ''
        ]
    ]

    id_table = Table(id_data, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    id_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LN_LIGHT_GREY),
        ('BACKGROUND', (0, 2), (-1, 2), LN_LIGHT_GREY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, 0), LN_MID_GREY),
        ('TEXTCOLOR', (0, 2), (-1, 2), LN_MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, LN_MID_GREY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LN_LIGHT_GREY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(id_table)
    elements.append(Spacer(1, 4*mm))

    # PEP & Sanctions
    elements.append(Paragraph("PEP & Sanctions Screening", styles['FieldLabel']))

    pep_sanctions_data = [
        ['PEP Match', 'PEP Confidence', 'Sanctions Match', 'Sanctions Confidence'],
        [
            'Yes' if ra.pep_match else 'No',
            str(ra.pep_confidence) + '%' if ra.pep_confidence else 'N/A',
            'Yes' if ra.sanctions_match else 'No',
            str(ra.sanctions_confidence) + '%' if ra.sanctions_confidence else 'N/A'
        ]
    ]

    pep_table = Table(pep_sanctions_data, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    pep_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LN_LIGHT_GREY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, 0), LN_MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, LN_MID_GREY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LN_LIGHT_GREY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(pep_table)
    elements.append(Spacer(1, 3*mm))

    # PEP evidence if found
    if ra.pep_match and ra.pep_evidence:
        elements.append(Paragraph("PEP Evidence:", styles['FieldLabel']))
        elements.append(Paragraph(f"Source: {ra.pep_evidence.source_list}", styles['BulletItem']))
        elements.append(Paragraph(f"Match Record: {ra.pep_evidence.match_record}", styles['BulletItem']))
        elements.append(Paragraph(f"Match Basis: {ra.pep_evidence.match_basis}", styles['BulletItem']))
        elements.append(Spacer(1, 3*mm))

    # Sanctions evidence if found
    if ra.sanctions_match and ra.sanctions_evidence:
        elements.append(Paragraph("Sanctions Evidence:", styles['FieldLabel']))
        elements.append(Paragraph(f"Source: {ra.sanctions_evidence.source_list}", styles['BulletItem']))
        elements.append(Paragraph(f"Match Record: {ra.sanctions_evidence.match_record}", styles['BulletItem']))
        elements.append(Paragraph(f"Match Basis: {ra.sanctions_evidence.match_basis}", styles['BulletItem']))
        elements.append(Spacer(1, 3*mm))

    # Adverse media
    elements.append(Paragraph(
        f"Adverse Media: {ra.adverse_media_count} article(s) found",
        styles['FieldLabel']
    ))
    if ra.adverse_media_count > 0:
        for article in ra.adverse_media_articles:
            elements.append(Paragraph(
                f"• {article.description} ({article.evidence_type})",
                styles['BulletItem']
            ))
    else:
        elements.append(Paragraph("No adverse media identified.", styles['BodyText2']))
    elements.append(Spacer(1, 4*mm))

    # TraceIQ findings
    elements.append(Paragraph("TraceIQ Findings", styles['FieldLabel']))
    trace = ra.trace_iq

    traceiq_data = [
        ['LexID', 'Current Address', 'Residency Score', 'Last Seen'],
        [
            str(trace.lex_id) if trace.lex_id else 'N/A',
            trace.current_address or 'N/A',
            trace.residency_score or 'N/A',
            str(trace.last_seen) if trace.last_seen else 'N/A'
        ],
        ['Mortality Status', 'Bankruptcy/IVA/DRO', 'CCJ Records', 'Sources Count'],
        [
            trace.mortality_status,
            'Yes' if trace.has_bankruptcy_iva_dro else 'No',
            str(len(trace.ccj_records)) + ' found',
            str(trace.sources_count)
        ]
    ]

    traceiq_table = Table(traceiq_data, colWidths=[40*mm, 50*mm, 35*mm, 35*mm])
    traceiq_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LN_LIGHT_GREY),
        ('BACKGROUND', (0, 2), (-1, 2), LN_LIGHT_GREY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, 0), LN_MID_GREY),
        ('TEXTCOLOR', (0, 2), (-1, 2), LN_MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 0.5, LN_MID_GREY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LN_LIGHT_GREY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))
    elements.append(traceiq_table)
    elements.append(Spacer(1, 4*mm))

    # Detailed analysis
    elements.append(Paragraph("Detailed Analysis (Agent 1)", styles['FieldLabel']))
    elements.append(Paragraph(ra.detailed_analysis, styles['BodyText2']))

    elements.append(section_divider())
    return elements


def build_audit_section(report: FinalReport, styles) -> list:
    """Builds the audit trail reference section."""
    elements = []

    elements.append(Paragraph("4. AUDIT TRAIL & SYSTEM INFORMATION", styles['SectionHeading']))
    elements.append(orange_divider())

    audit_data = [
        ['Case ID:', report.case_id],
        ['Customer Name:', report.customer_name],
        ['Report Timestamp:', report.report_timestamp[:19].replace('T', ' ')],
        ['Audit Trail Reference:', report.audit_trail_reference or 'N/A'],
        ['System:', 'LexisNexis Agentic Identity Risk Management System'],
        ['Model:', 'Llama-3.3-70B-Instruct via HuggingFace Inference API'],
        ['Pipeline:', 'Agent 1 → Routing → Agent 2 → Agent 3 → FCA Recheck'],
    ]

    audit_table = Table(audit_data, colWidths=[50*mm, 110*mm])
    audit_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), LN_MID_GREY),
        ('TEXTCOLOR', (1, 0), (1, -1), LN_DARK),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LN_WHITE, LN_LIGHT_GREY]),
        ('BOX', (0, 0), (-1, -1), 0.5, LN_MID_GREY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LN_LIGHT_GREY),
    ]))
    elements.append(audit_table)
    elements.append(Spacer(1, 6*mm))

    # Confidentiality notice
    elements.append(Paragraph(
        "CONFIDENTIALITY NOTICE: This report contains sensitive personal data processed in accordance with "
        "UK GDPR and the Data (Use and Access) Act 2025. It is intended solely for the named Senior Analyst "
        "and must not be shared, forwarded, or disclosed to any unauthorised party. Unauthorised disclosure "
        "may constitute a breach of data protection law.",
        ParagraphStyle(
            'ConfidentialityNotice',
            fontName='Helvetica-Oblique',
            fontSize=7,
            textColor=LN_MID_GREY,
            spaceAfter=4,
            leading=10
        )
    ))

    return elements


def generate_report_pdf(report: FinalReport, output_dir: str = "reports") -> str:
    """
    Main function — generates the complete PDF report from a FinalReport object.
    Saves the PDF to the output directory and returns the file path.

    Args:
        report: The complete FinalReport object from the pipeline
        output_dir: Directory where the PDF will be saved (default: "reports")

    Returns:
        The full file path of the generated PDF
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build the output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Report_{report.case_id}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Set up the document with A4 page size and margins
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
        title=f"Identity Risk Assessment Report — {report.case_id}",
        author="LexisNexis Agentic Identity Risk Management System",
        subject=f"Risk Assessment for {report.customer_name}",
    )

    # Build styles
    styles = build_styles()

    # Build all sections in sequence
    story = []
    story.extend(build_cover_page(report, styles))
    story.extend(build_executive_summary(report, styles))
    story.extend(build_compliance_section(report, styles))
    story.extend(build_risk_assessment_section(report, styles))
    story.extend(build_audit_section(report, styles))

    # Generate the PDF
    doc.build(story)

    print(f"[PDF REPORT] Generated: {filepath}")
    return filepath