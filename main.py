"""
main.py - Orchestration Layer

Connects Agent 1 (Risk Analysis) -> Agent 2 (Compliance) -> Agent 3 (Report Generation) using Langgraph.
Handles deterministic routing logic and audit logging.

"""

from langgraph.graph import StateGraph, START, END
from state import IdentityRiskState
from agents.agent1_risk import run_agent1
from agents.agent2_compliance import run_agent2
from audit_logger import create_audit_entry, write_audit_trail
from dotenv import load_dotenv
import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


## Agent 1 Node:

def agent1_node(state: IdentityRiskState) -> IdentityRiskState:
    file_path = state["file_path"]
    # TODO: wrap run_agent1() in try/except to log failures too, not just
    # successes - currently, if Agent 1 crashes, nothing gets recorded in
    # audit_log about the failure, which is itself a real accountability gap
    risk_assessment = run_agent1(file_path)

    state["risk_assessment"] = risk_assessment
    state["audit_log"].append(                    # a Python list can hold any type of item inside it, including dictionaries: (audit_log is a list OF dictionaries)
        create_audit_entry(
            case_id=risk_assessment.case_id,
            step="Agent 1 - Risk Analysis",
            action="Completed risk assessment",
            details={
                "overall_risk_rating": risk_assessment.overall_risk_rating,
                "pep_match": risk_assessment.pep_match,
                "mortality_status": risk_assessment.trace_iq.mortality_status

            }
        )
    )

    return state


# -------------------------------------------------------------------- Routing Logics ------------------------------------------------------------------------------------

def routing_node(state: IdentityRiskState) -> IdentityRiskState:
    """
    Routing Logic Node - Deterministic Overrides
    ----------------------------------------------------------------------
    Agent 1's LLM produces its own guess for requires_human_review and enhanced_due_diligence_required, but this guess is unreliable - testing
    showed the same case can produce different values (True/False) across repeated runs of the identical prompt and data. This node OVERRIDES
    those unreliable LLM guesses with fixed, deterministic formulas, so downstream decisions (Agent 2's compliance rules) are based on
    consistent, reproducible logic rather than LLM judgement variability.
    """

    risk_assessment = state["risk_assessment"]

    # ------------------------------------------------------------------
    # requires_human_review, Connects to: GDPR - Data (Use and Access) Act 2025, Article 22C.
    # Article 22C requires that solely automated decisions with significant effects on a person be accompanied by safeguards (information,
    # right to respond, human intervention, right to contest). Whether a human reviews this case is the deciding factor for whether this
    # system currently satisfies Article 22C (see Agent 2, Rule 3). Threshold: confidence_score < 70 triggers human review - i.e. if
    # Agent 1's own stated confidence in its risk assessment is low, a human should check the case rather than trust an uncertain automated conclusion.
    # ------------------------------------------------------------------

    risk_assessment.requires_human_review = risk_assessment.confidence_score < 70


    # ------------------------------------------------------------------
    # enhanced_due_diligence_required, Connects to: AML/KYC - Money Laundering, Terrorist Financing and Transfer of Funds (Information on the Payer) Regulations 2017,
    # Regulation 33 (Enhanced Due Diligence, general high-risk trigger) and Regulation 35 (mandatory EDD specifically for PEPs).
    # Threshold: risk_score > 75 (Reg. 33's general high-risk trigger) OR pep_match is True (Reg. 35's mandatory PEP trigger) - either
    # condition alone is sufficient to require EDD.
    # ------------------------------------------------------------------

    risk_assessment.enhanced_due_diligence_required = (
        risk_assessment.risk_score > 75 or risk_assessment.pep_match
    )


    # ==============================================================  Temporary =====================================================================

    # TEMPORARY placeholder for human analyst confirmation of EDD completion.
    # This simulates the manual confirmation step that will eventually be
    # replaced by the real notification/reviewer system (see TODO in schemas.py).
    # Only asks if EDD is actually required - otherwise stays None (not applicable).
    if risk_assessment.enhanced_due_diligence_required:
        response = input(
            f"\n[MANUAL CONFIRMATION NEEDED] EDD is required for case "
            f"{risk_assessment.case_id}. Has EDD actually been completed? "
            f"(yes/no/skip): "
        ).strip().lower()

        if response == "yes":
            risk_assessment.enhanced_due_diligence_completed = True
        elif response == "no":
            risk_assessment.enhanced_due_diligence_completed = False
        # "skip" or anything else leaves it as None (not yet confirmed)



    # =================================================================================================================================================



    state["risk_assessment"] = risk_assessment

    # -------------------------------------------------------------------

    state["audit_log"].append(
        create_audit_entry(
            case_id=risk_assessment.case_id,
            step="Routing Logic",
            action="Applied deterministic overrides (GDPR Article 22C; AML/KYC Reg. 33 & 35)",
            details={
                "requires_human_review": risk_assessment.requires_human_review,
                "enhanced_due_diligence_required": risk_assessment.enhanced_due_diligence_required,
            }
        )
    )

    return state



## Agent 2 Node:

def agent2_node(state: IdentityRiskState) -> IdentityRiskState:

    # All compliance rules (mortality, GDPR, FCA, AML/KYC) are enforced inside Agent 2's prompt (agent2_compliance.py, create_prompt()),
    # tested and verified across multiple cases. This node's job is purely orchestration - calling Agent 2 and logging the result.

    risk_assessment = state["risk_assessment"]
    compliance_status = run_agent2(risk_assessment)
    state["compliance_status"] = compliance_status
    state["audit_log"].append(
        create_audit_entry(
            case_id=compliance_status.case_id,
            step="Agent 2 - Compliance",
            action="Completed compliance check",
            details={
                "gdpr_compliant": compliance_status.gdpr_compliant,
                "fca_compliant": compliance_status.fca_compliant,
                "aml_kyc_compliant": compliance_status.aml_kyc_compliant,
                "mortality_confirmed_block": compliance_status.mortality_confirmed_block,
            }
        )
    )

    return state



## LangGraph Construction: Connects agent1 -> routing -> agent2, in sequence, using IdentityRiskState as the shared data that flows between each step.
#-----------------------------------------------------------------------------------------------------------------------------------------------------


def build_graph():

    # graph with shared state
    graph = StateGraph(IdentityRiskState)

    # Adding Nodes
    graph.add_node("agent1", agent1_node)
    graph.add_node("routing", routing_node)
    graph.add_node("agent2", agent2_node)

    # Connecting Edges
    graph.add_edge(START, "agent1")
    graph.add_edge("agent1", "routing")
    graph.add_edge("routing", "agent2")
    graph.add_edge("agent2", END)

    # Comile the graph
    workflow = graph.compile()

    return workflow


# -------------------------------------------------------------------- Running the Pipeline -----------------------------------------------------------------

def run_pipeline(file_path: str):
    workflow = build_graph()

    initial_state = {
        "file_path": file_path,
        "risk_assessment": None,
        "compliance_status": None,
        "final_report": None,
        "audit_log": []
    }

    final_state = workflow.invoke(initial_state)

    audit_filepath = write_audit_trail(
        audit_log=final_state["audit_log"],
        case_id=final_state["risk_assessment"].case_id
    )

    return final_state, audit_filepath                  # the file path where the audit trail JSON was actually saved

    # final_state is the fully updated version of the state, after passing through all 3 nodes (agent1 → routing → agent2) 
    # in sequence, containing everything each step contributed along the way.


if __name__ == "__main__":                                                 # this code only runs if you execute main.py directly
    FILE_PATH = os.path.join(BASE_DIR, "data", "CASE-2026-316.pdf")        # builds the full path to a specific test case file, using BASE_DIR (project's root folder) to construct it correctly.
    final_state, audit_filepath = run_pipeline(FILE_PATH)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Risk Assessment: {final_state['risk_assessment']}")            # the complete RiskAssessment object (Agent 1's output, after routing_node's corrections)
    print(f"\nCompliance Status: {final_state['compliance_status']}")      # the complete ComplianceStatus object (Agent 2's output)
    print(f"\nAudit trail saved to: {audit_filepath}")                     # file path where the audit trail JSON was actually saved.



