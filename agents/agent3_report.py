"""
## Agent 3: Report Generation Agent

Synthesizes Agent 1's RiskAssessment and Agent 2's ComplianceStatus into a final report with a coherent decision rationale.
Only generates the ReportSynthesis fields via LLM - everything else is assembled from data already known.
Single entry point: run_agent3(risk_assessment, compliance_status) -> FinalReport

"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from schemas import ReportSynthesis, FinalReport, RiskAssessment, ComplianceStatus
from datetime import datetime
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_prompt():
    parser = PydanticOutputParser(pydantic_object=ReportSynthesis)
    format_instructions = parser.get_format_instructions()

    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert report writer synthesising identity risk
         and compliance findings into a clear, coherent final report for a human
         analyst. You connect the risk analysis and compliance findings into one
         unified narrative, explaining WHY the final decision was reached - not
         just restating each agent's findings separately."""),

        ("human", """Based on the following risk assessment and compliance findings,
         synthesise a final report.

        Risk Assessment Summary:
        {risk_summary}

        Compliance Status Summary:
        {compliance_summary}

        1. Write a brief executive summary (2-3 sentences) covering the case's
           overall outcome. This summary MUST briefly mention the compliance
           status of ALL THREE areas - GDPR, FCA, AND AML/KYC - EXACTLY as
           given in the Compliance Status Summary above (gdpr_compliant,
           fca_compliant, aml_kyc_compliant) - do NOT state a compliance
           status opposite to what is actually given. For each area, include
           a SHORT reason in parentheses (a few words, not a full sentence).
           This reason MUST be taken from the CONTENT of that area's own
           Compliance Flags entry (e.g. "detailed analysis satisfies FCA
           principles" or "no PEP match, no EDD required") - it must NOT be
           "no action required" or similar text taken from
           Regulatory Actions Required, since that describes the remediation
           action, not the underlying reason. Do NOT invent or guess a
           reason, and do NOT borrow a reason from a different compliance
           area. This summary MUST also be factually consistent with the
           Enhanced Due Diligence status stated later in this report - if EDD
           is not required, do NOT mention it as required or needed anywhere
           in this summary.
        2. State the final overall risk rating.
        3. Write a decision rationale that CONNECTS the risk findings and
           compliance findings into one coherent explanation - explicitly
           reference specific risk signals AND specific compliance flags
           together, showing how they combine to support the final recommendation.
           When explaining GDPR compliance status, use this EXACT, unambiguous
           structure, with no self-correction or contradictory clauses: state
           whether a human is reviewing the case, then separately state the
           Confidence Score value compared against the threshold of 70 as the
           reason (e.g. "No human is reviewing this case, as Confidence Score X
           met the threshold of 70. This means Article 22C safeguards are
           required but not currently in place, making this case GDPR
           non-compliant."). Do not write a sentence that states one thing then
           reverses or corrects itself mid-sentence.
           IMPORTANT: only connect facts that are genuinely causally related - do
           not invent a connection between unrelated facts to make the narrative
           flow better (e.g. the liveness check confirms a live person is present,
           NOT residency or address - do not conflate these). State each distinct
           fact accurately on its own terms, and only draw connections that are
           actually true.
           This is the most important field - do not just list findings
           separately, synthesise them into a genuine narrative.
        IMPORTANT: Always explicitly state whether Enhanced Due Diligence is
        required for this case, AND WHY, based on risk_score and PEP Match
        (e.g. "Enhanced Due Diligence: Required, because PEP Match is True" or
        "Enhanced Due Diligence: Required, because risk score exceeded 75" or
        "Enhanced Due Diligence: Not required, as risk score did not exceed 75
        and no PEP match was found"). If Required is True AND Completed is
        True, also state that it has been confirmed completed. If Required is
        True and Completed is False or None, do not state "not completed" as
        a separate fact - instead, naturally include completing it as part of
        the recommendations/next steps. If Required is False, do NOT
        recommend enhanced due diligence anywhere in the report, even if
        Agent 1's own risk narrative suggests it - the deterministic Required
        field is authoritative. Additionally, do NOT use vague phrases like
        "additional due diligence" or "further due diligence" anywhere in the
        report, as these could be confused with Enhanced Due Diligence
        specifically - use precise language instead (e.g. "gather additional
        identity documents," "conduct further identity checks").
        IMPORTANT: Check GDPR compliance status above. If gdpr_compliant is True
        (a human IS reviewing this case, Article 22C safeguards are already
        satisfied), do NOT recommend "implement human review" or "implement
        Article 22C safeguards" anywhere in the report - these are already in
        place. Only recommend implementing these safeguards if gdpr_compliant
        is False (no human review, safeguards genuinely missing).
        4. List final recommendations for the analyst.
        5. List final next steps.

        {format_instructions}""")
    ])

    return report_prompt.partial(format_instructions=format_instructions)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Agent 3 writes the report before we know the FINAL FCA verdict. Later, that verdict gets corrected. This function goes back and fixes just the FCA part of the report so it matches the correction.

"""
def create_report_revision_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You revise two parts of a compliance report to ensure they
         accurately reflect the FINAL, confirmed compliance status - correcting
         any outdated or contradictory statements about FCA compliance
         specifically, while keeping everything else unchanged."""),

        ("human", """Here is the current executive summary:

        {executive_summary}

        Here is the current decision rationale:

        {decision_rationale}

        Here are the current final recommendations:

        {final_recommendations}

        Here are the current final next steps:

        {final_next_steps}

        The FINAL, confirmed FCA compliance status is: {fca_compliant_final}
        Reason: {fca_reason}

        Rewrite ALL FOUR sections so that any statement about FCA compliance
        accurately reflects this FINAL status - correct any contradictory or
        outdated FCA-related statements in ANY section. Specifically:
        - If fca_compliant_final is True: REMOVE any recommendation or next
          step suggesting FCA remediation (e.g. "provide more detailed
          analysis for FCA", "improve FCA explainability") - these are
          incorrect if FCA is compliant.
        - If fca_compliant_final is False: ENSURE recommendations include
          appropriate FCA remediation steps.
        - Keep all other content (risk findings, GDPR, AML/KYC) exactly
          as it is.

        Return your response in EXACTLY this format, nothing else:
        EXECUTIVE_SUMMARY: [revised executive summary here]
        DECISION_RATIONALE: [revised decision rationale here]
        FINAL_RECOMMENDATIONS: [revised final recommendations here, as a
        bullet list with each item on a new line starting with "- "]
        FINAL_NEXT_STEPS: [revised final next steps here, as a bullet list
        with each item on a new line starting with "- "]""")
    ])
    return prompt


# Actually calls the LLM to do the rewrite described above

def run_report_revision(executive_summary: str, decision_rationale: str,
                        final_recommendations: list, final_next_steps: list,
                        fca_compliant_final: bool, fca_reason: str):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        max_new_tokens=1000
    )
    model = ChatHuggingFace(llm=llm)

    prompt = create_report_revision_prompt()


    chain = prompt | model

    # Convert lists to bullet format for the prompt
    recommendations_text = "\n".join(f"- {r}" for r in final_recommendations)
    next_steps_text = "\n".join(f"- {s}" for s in final_next_steps)

    result = chain.invoke({                                  # result is an AIMessage - we only need its .content (the plain revised text)
        "executive_summary": executive_summary,
        "decision_rationale": decision_rationale,
        "final_recommendations": recommendations_text,
        "final_next_steps": next_steps_text,
        "fca_compliant_final": fca_compliant_final,
        "fca_reason": fca_reason
    })

    # Parse the 4-section format the LLM was asked to return
    text = result.content

    revised_summary = text.split("EXECUTIVE_SUMMARY:")[1].split("DECISION_RATIONALE:")[0].strip()
    revised_rationale = text.split("DECISION_RATIONALE:")[1].split("FINAL_RECOMMENDATIONS:")[0].strip()
    revised_recommendations_text = text.split("FINAL_RECOMMENDATIONS:")[1].split("FINAL_NEXT_STEPS:")[0].strip()
    revised_next_steps_text = text.split("FINAL_NEXT_STEPS:")[1].strip()


    # Convert bullet text back to lists
    revised_recommendations = [
        line.lstrip("- ").strip()
        for line in revised_recommendations_text.split("\n")
        if line.strip() and line.strip() != "-"
    ]

    revised_next_steps = [
        line.lstrip("- ").strip()
        for line in revised_next_steps_text.split("\n")
        if line.strip() and line.strip() != "-"
    ]

    return revised_summary, revised_rationale, revised_recommendations, revised_next_steps

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_report_generation(risk_assessment: RiskAssessment, compliance_status: ComplianceStatus) -> ReportSynthesis:

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        max_new_tokens=2000
    )
    model = ChatHuggingFace(llm=llm)

    parser = PydanticOutputParser(pydantic_object=ReportSynthesis)
    report_prompt = create_prompt()


    risk_summary = f"""
    Case ID: {risk_assessment.case_id}
    Customer Name: {risk_assessment.customer_name}
    Overall Risk Rating: {risk_assessment.overall_risk_rating}
    Key Signals: {', '.join(risk_assessment.key_signals)}
    Detailed Analysis: {risk_assessment.detailed_analysis}
    Recommended Action: {risk_assessment.recommended_action}
    Enhanced Due Diligence Required: {risk_assessment.enhanced_due_diligence_required}
    Enhanced Due Diligence Completed: {risk_assessment.enhanced_due_diligence_completed}
    """

    compliance_summary = f"""
    GDPR Compliant: {compliance_status.gdpr_compliant}
    FCA Compliant: {compliance_status.fca_compliant}
    AML/KYC Compliant: {compliance_status.aml_kyc_compliant}
    Mortality Confirmed Block: {compliance_status.mortality_confirmed_block}
    Compliance Flags: {', '.join(compliance_status.compliance_flags)}
    Regulatory Actions Required: {', '.join(compliance_status.regulatory_actions_required)}
    """

    chain = report_prompt | model | parser

    result = chain.invoke({
        "risk_summary": risk_summary,
        "compliance_summary": compliance_summary
    })

    return result



def run_agent3(risk_assessment: RiskAssessment, compliance_status: ComplianceStatus, audit_trail_reference: str = "") -> FinalReport:
    synthesis = run_report_generation(risk_assessment, compliance_status)

    final_report = FinalReport(
        case_id=risk_assessment.case_id,
        customer_name=risk_assessment.customer_name,
        executive_summary=synthesis.executive_summary,
        overall_risk_rating=synthesis.overall_risk_rating,
        decision_rationale=synthesis.decision_rationale,
        final_recommendations=synthesis.final_recommendations,
        final_next_steps=synthesis.final_next_steps,
        risk_assessment=risk_assessment,
        compliance_status=compliance_status,
        audit_trail_reference=audit_trail_reference,
        report_timestamp=datetime.now().isoformat()
    )

    return final_report


# ------------------------------------------------------------------------------------

def print_readable_report(final_report: FinalReport):
    """Clean, readable printout of the FinalReport, split by section."""
    print(f"\nCase ID: {final_report.case_id}")
    print(f"Customer: {final_report.customer_name}")
    print(f"\nExecutive Summary:\n{final_report.executive_summary}")
    print(f"\nOverall Risk Rating: {final_report.overall_risk_rating}")
    print(f"\nDecision Rationale:\n{final_report.decision_rationale}")
    print(f"\nFinal Recommendations:")
    for rec in final_report.final_recommendations:
        print(f"  - {rec}")
    print(f"\nFinal Next Steps:")
    for step in final_report.final_next_steps:
        print(f"  - {step}")

    print("\n" + "-"*80)
    print("RISK ASSESSMENT (Agent 1 full output):")
    print("-"*80)
    print(final_report.risk_assessment)

    print("\n" + "-"*80)
    print("COMPLIANCE STATUS (Agent 2 full output):")
    print("-"*80)
    print(final_report.compliance_status)

    print("\n" + "-"*80)
    print(f"Audit Trail Reference: {final_report.audit_trail_reference}")
    print(f"Report Timestamp: {final_report.report_timestamp}")
    print("-"*80 + "\n")

# -------------------------------------------------------------------------------------


if __name__ == "__main__":
    from agents.agent1_risk import run_agent1
    from agents.agent2_compliance import run_agent2

    FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-303.pdf')
    risk_assessment = run_agent1(FILE_PATH)

    compliance_status = run_agent2(risk_assessment)


    final_report = run_agent3(risk_assessment, compliance_status)

    print("\n" + "="*80)
    print("AGENT 3 OUTPUT (Final Report)")
    print("="*80)
    # print(final_report)

    print_readable_report(final_report)