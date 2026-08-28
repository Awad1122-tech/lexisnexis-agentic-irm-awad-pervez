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
           overall outcome.
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
        4. List final recommendations for the analyst.
        5. List final next steps.

        {format_instructions}""")
    ])

    return report_prompt.partial(format_instructions=format_instructions)


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