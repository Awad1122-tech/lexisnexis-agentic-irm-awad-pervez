"""
main.py - Orchestration Layer

Connects Agent 1 (Risk Analysis) -> Routing -> Agent 2 (Compliance) -> Agent 3 (Report Generation) -> FCA Recheck using LangGraph.
Handles deterministic routing logic, reviewer notifications, secure dashboard reviews, and audit logging.

"""

from langgraph.graph import StateGraph, START, END
from state import IdentityRiskState
from agents.agent1_risk import run_agent1
from agents.agent2_compliance import (
    run_agent2, run_fca_final_check,
    load_regulatory_docs, split_documents,
    create_regulatory_vector_store, create_retriever
)
from agents.agent3_report import run_agent3, run_report_revision

from reviewer_app.app import (
    start_dashboard,                # starts the Flask web server in a background thread  (thread: a background worker that runs alongside the main program (pipeline) simultaneously) / Second worker (Flask/thread): stands at the door, waiting for the reviewer to arrive, runs quietly in the background
    register_review,                # stores a new review session (token + PIN + case details) in memory, so Flask knows what to show when the reviewer opens the link
    get_decision,                   # retrieves what the reviewer clicked after they responded
    wait_for_decision,              # pauses the pipeline until the reviewer makes a decision
    pending_reviews                 # imported directly so main.py can delete tokens after retrieving decisions
)

from reviewer_app.notifier import (
    generate_token,                   # creates a unique random string embedded in the dashboard URL, acts as a one-time password
    generate_pin,                     # creates a 6-digit number sent in a separate email, second security layer
    send_human_review_notification,   # sends two emails (link + PIN) when confidence score is below 70
    send_edd_notification,             # sends two emails (link + PIN) when EDD confirmation is required
    send_senior_analyst_report    
)

from reviewer_app.report_pdf import generate_report_pdf

import webbrowser  # built-in Python module, opens URLs in the browser

from audit_logger import create_audit_entry, write_audit_trail
from dotenv import load_dotenv
import datetime
import os


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================================================================================

# Dashboard base URL - where Flask is running
# In production this would be a real domain (e.g. https://reviewer.company.com)
# For now it runs locally on port 5000

DASHBOARD_BASE_URL = "http://localhost:5000"   # base address of the Flask dashboard running locally, combined with the token to build the full review URL


# ==========================================================================================================================================




## Agent 1 Node:

def agent1_node(state: IdentityRiskState) -> IdentityRiskState:
    file_path = state["file_path"]

    try:
        risk_assessment = run_agent1(file_path)  # Attempt to run Agent 1, if it succeeds, great. If anything goes wrong, jump to except.

    except Exception as e:   # Catches any type of error that could occur, e is the actual error object containing the error message (Exception is the base class for all errors in Python.)
        # if ANYTHING goes wrong (API down, PDF corrupted, parsing failure etc.)
        # log the failure to the audit trail before stopping the pipeline
        print(f"[AGENT 1] Failed: {e}")
        state["audit_log"].append(
            create_audit_entry(
                case_id="UNKNOWN",  # case ID not yet known since Agent 1 failed
                step="Agent 1 - Risk Analysis",
                action="FAILED — exception during risk assessment",
                details={"error": str(e)}  # record exactly what went wrong
            )
        )
        raise  # re-raise the error so LangGraph knows the pipeline failed

    # only reached if Agent 1 succeeded

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


    # ==============================================================  Review Notification Start =====================================================================

    # Start Flask dashboard in background thread
    # Only starts once,  ignored if already running

    start_dashboard(port=5000)                          # starts the Flask web server in a background thread on port 5000,  called once, ignored if already running

    # ----------------------------------------------------------
    # HUMAN REVIEW NOTIFICATION (GDPR Article 22C)
    # Triggered when confidence score is below 70
    # ----------------------------------------------------------

    if risk_assessment.requires_human_review:

        print(f"\n[ROUTING] Low confidence score "
              f"({risk_assessment.confidence_score}) — "
              f"human review required (GDPR Article 22C)")

        hr_token = generate_token()
        hr_pin = generate_pin()
        hr_url = f"{DASHBOARD_BASE_URL}/review?token={hr_token}"

        register_review(
            token=hr_token,
            case_id=risk_assessment.case_id,
            customer_name=risk_assessment.customer_name,
            review_type="human_review",
            pin=hr_pin,
            risk_assessment=risk_assessment
        )

        send_human_review_notification(
            case_id=risk_assessment.case_id,
            customer_name=risk_assessment.customer_name,
            dashboard_url=hr_url,
            pin=hr_pin
        )

        print(f"[ROUTING] Opening reviewer dashboard: {hr_url}")
        webbrowser.open(hr_url)

        responded = wait_for_decision(timeout_seconds=86400)  # pauses the pipeline here and waits up to 24 hours (86400 seconds) for the reviewer to respond,  returns True if reviewer responded, False if timeout expired

        if responded:                                         # True = reviewer clicked a button within 24 hours
            hr_decision = get_decision(hr_token)              # retrieve what the reviewer clicked ("acknowledged")
            del pending_reviews[hr_token]                     # token now permanently invalid, link can never be reused

            print(f"[ROUTING] Human review decision: {hr_decision}")  # log decision to terminal
            state["audit_log"].append(                                # record this event in the audit trail
                create_audit_entry(
                    case_id=risk_assessment.case_id,
                    step="Routing Logic - Human Review",
                    action="Human review completed by reviewer",
                    details={
                        "decision": hr_decision,
                        "confidence_score": risk_assessment.confidence_score,
                        "gdpr_article_22c": "Reviewer acknowledged"
                    }
                )
            )
        else:     # False = 24 hours passed with no reviewer response
            print("[ROUTING] Human review timeout — no response received.")  # log timeout to terminal
            state["audit_log"].append(                                       # record the timeout in audit trail,  missed review is a compliance event
                create_audit_entry(
                    case_id=risk_assessment.case_id,
                    step="Routing Logic - Human Review",
                    action="Human review TIMED OUT — no reviewer response",
                    details={
                        "confidence_score": risk_assessment.confidence_score,
                        "timeout_hours": 24
                    }
                )
            )

    # ----------------------------------------------------------------------------------------------------

    # EDD CONFIRMATION NOTIFICATION (AML Reg. 33 & 35)
    # Triggered when risk_score > 75 OR pep_match is True

    # ----------------------------------------------------------------------------------------------------

    if risk_assessment.enhanced_due_diligence_required:

        print(f"\n[ROUTING] EDD required "
              f"(risk_score={risk_assessment.risk_score}, "
              f"pep_match={risk_assessment.pep_match})")

        edd_token = generate_token()
        edd_pin = generate_pin()
        edd_url = f"{DASHBOARD_BASE_URL}/review?token={edd_token}"

        register_review(
            token=edd_token,
            case_id=risk_assessment.case_id,
            customer_name=risk_assessment.customer_name,
            review_type="edd",
            pin=edd_pin,
            risk_assessment=risk_assessment
        )

        send_edd_notification(
            case_id=risk_assessment.case_id,
            customer_name=risk_assessment.customer_name,
            dashboard_url=edd_url,
            pin=edd_pin
        )

        print(f"[ROUTING] Opening EDD dashboard: {edd_url}")
        webbrowser.open(edd_url)

        responded = wait_for_decision(timeout_seconds=86400)

        if responded:
            edd_decision = get_decision(edd_token)
            del pending_reviews[edd_token]  # token now permanently invalid
            print(f"[ROUTING] EDD decision: {edd_decision}")

            if edd_decision == "yes":
                risk_assessment.enhanced_due_diligence_completed = True
            elif edd_decision == "no":
                risk_assessment.enhanced_due_diligence_completed = False
            else:
                risk_assessment.enhanced_due_diligence_completed = None

            state["audit_log"].append(
                create_audit_entry(
                    case_id=risk_assessment.case_id,
                    step="Routing Logic - EDD Confirmation",
                    action="EDD confirmation received from reviewer",
                    details={
                        "decision": edd_decision,
                        "enhanced_due_diligence_completed":
                            risk_assessment.enhanced_due_diligence_completed,
                        "trigger": "PEP match" if risk_assessment.pep_match
                                   else f"High risk score ({risk_assessment.risk_score})"
                    }
                )
            )
        else:
            print("[ROUTING] EDD timeout — no response received.")
            risk_assessment.enhanced_due_diligence_completed = None
            state["audit_log"].append(
                create_audit_entry(
                    case_id=risk_assessment.case_id,
                    step="Routing Logic - EDD Confirmation",
                    action="EDD confirmation TIMED OUT — no reviewer response",
                    details={
                        "timeout_hours": 24,
                        "enhanced_due_diligence_completed": None
                    }
                )
            )


    # ==============================================================  Review Notification End =====================================================================



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

    try:
        # attempt to run Agent 2 normally
        compliance_status = run_agent2(risk_assessment)

    except Exception as e:
        # if anything goes wrong (API down, RAG retrieval failure, parsing failure etc.)
        # log the failure to the audit trail before stopping the pipeline
        print(f"[AGENT 2] Failed: {e}")
        state["audit_log"].append(
            create_audit_entry(
                case_id=risk_assessment.case_id,  # case ID known since Agent 1 already succeeded
                step="Agent 2 - Compliance",
                action="FAILED - exception during compliance assessment",
                details={"error": str(e)}  # record exactly what went wrong
            )
        )
        raise  # re-raise the error so LangGraph knows the pipeline failed


    # only reached if Agent 2 succeeded
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


## Agent 3 Node:

def agent3_node(state: IdentityRiskState) -> IdentityRiskState:
    risk_assessment = state["risk_assessment"]
    compliance_status = state["compliance_status"]

    try:
        # attempt to run Agent 3 normally
        final_report = run_agent3(risk_assessment, compliance_status)

    except Exception as e:
        # if ANYTHING goes wrong (API down, LLM synthesis failure, parsing failure etc.)
        # log the failure to the audit trail before stopping the pipeline
        print(f"[AGENT 3] Failed: {e}")
        state["audit_log"].append(
            create_audit_entry(
                case_id=risk_assessment.case_id,  # case ID known since Agent 1 already succeeded
                step="Agent 3 - Report Generation",
                action="FAILED — exception during report synthesis",
                details={"error": str(e)}  # record exactly what went wrong
            )
        )
        raise  # re-raise the error so LangGraph knows the pipeline failed


    # only reached if Agent 3 succeeded
    state["final_report"] = final_report
    state["audit_log"].append(
        create_audit_entry(
            case_id=final_report.case_id,
            step="Agent 3 - Report Generation",
            action="Completed final report synthesis",
            details={
                "decision_rationale": final_report.decision_rationale,
                "final_recommendations": final_report.final_recommendations
            }
        )
    )

    return state


## FCA_recheck_node:

def fca_recheck_node(state: IdentityRiskState) -> IdentityRiskState:
    final_report = state["final_report"]

    try:
        #----------------------------------------------------
        docs = load_regulatory_docs()
        chunks = split_documents(docs)
        vector_store = create_regulatory_vector_store(chunks)
        retriever = create_retriever(vector_store)
        #----------------------------------------------------

        fca_final_result = run_fca_final_check(
            retriever=retriever,
            detailed_analysis=final_report.risk_assessment.detailed_analysis,
            decision_rationale = final_report.decision_rationale,
            audit_log = state["audit_log"]
        )

    except Exception as e:
        # if ANYTHING goes wrong (API down, RAG retrieval failure, LLM failure etc.)
        # log the failure to the audit trail before stopping the pipeline
        print(f"[FCA RECHECK] Failed: {e}")
        state["audit_log"].append(
            create_audit_entry(
                case_id=final_report.case_id,  # case ID known since all previous nodes succeeded
                step="FCA Final Check",
                action="FAILED — exception during FCA re-evaluation",
                details={"error": str(e)}  # record exactly what went wrong
            )
        )
        raise  # re-raise the error so LangGraph knows the pipeline failed


    # only reached if FCA recheck succeeded
    final_report.compliance_status.fca_compliant = fca_final_result.fca_compliant_final

    # Always revise ALL FOUR fields to match the FINAL FCA verdict
    revised_summary, revised_rationale, revised_recommendations, revised_next_steps  = run_report_revision(
        executive_summary=final_report.executive_summary,
        decision_rationale=final_report.decision_rationale,
        final_recommendations=final_report.final_recommendations,
        final_next_steps=final_report.final_next_steps,
        fca_compliant_final=fca_final_result.fca_compliant_final,
        fca_reason=fca_final_result.reason
    )
    final_report.executive_summary = revised_summary
    final_report.decision_rationale = revised_rationale
    final_report.final_recommendations = revised_recommendations
    final_report.final_next_steps = revised_next_steps

    final_report.compliance_status.compliance_flags.append(
        f"FCA (final review, post-synthesis): {fca_final_result.reason}"  # add the final, authoritative FCA reasoning
    )

    # Fixes stale "FCA: No action required" left over from the early check -
    # rebuilds the FCA action item to match the FINAL, corrected verdict

    updated_actions = []                                                                        # build a fresh, corrected version of the actions list
    for action in final_report.compliance_status.regulatory_actions_required:
        if action.startswith("FCA:"):                                                           # find the one action item that's about FCA
            if fca_final_result.fca_compliant_final:                                            # True = FCA is compliant
                updated_actions.append("FCA: No action required (confirmed by final review).")
            else:                                                                               # False = FCA is non-compliant
                updated_actions.append(f"FCA: {fca_final_result.reason}")                       # add the FCA action using the Final Check's own reason as the explanation
        else:
            updated_actions.append(action)                                                      # not the FCA entry - keep it exactly as it was (GDPR or AML/KYC)
    final_report.compliance_status.regulatory_actions_required = updated_actions                # save the corrected list back into the final report

    state["final_report"] = final_report
    state["audit_log"].append(
        create_audit_entry(
            case_id=final_report.case_id,
            step="FCA Final Check",
            action="Completed post-synthesis FCA re-evaluation",
            details={
                "fca_compliant_final": fca_final_result.fca_compliant_final,
                "reason": fca_final_result.reason,
            }
        )
    )

    return state


## LangGraph Construction: Connects agent1 -> routing -> agent2 -> agent3 -> fca_recheck, in sequence, using IdentityRiskState as the shared data that flows between each step.
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def build_graph():

    # graph with shared state
    graph = StateGraph(IdentityRiskState)

    # Adding Nodes
    graph.add_node("agent1", agent1_node)
    graph.add_node("routing", routing_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("fca_recheck", fca_recheck_node)

    # Connecting Edges
    graph.add_edge(START, "agent1")
    graph.add_edge("agent1", "routing")
    graph.add_edge("routing", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_edge("agent3", "fca_recheck")
    graph.add_edge("fca_recheck", END)

    # Compile the graph
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

    # Now that the audit trail file actually exists, update the final report
    # with the real reference (it was an empty placeholder during Agent 3's run)
    final_state["final_report"].audit_trail_reference = audit_filepath



    # Generate the professional PDF report for the Senior Analyst
    pdf_filepath = generate_report_pdf(
        report=final_state["final_report"],
        output_dir="reports"  # saves to reports/ folder in project root
    )

    # Email the PDF report to the Senior Analyst
    send_senior_analyst_report(
        case_id=final_state["final_report"].case_id,
        customer_name=final_state["final_report"].customer_name,
        overall_risk_rating=final_state["final_report"].overall_risk_rating,
        pdf_filepath=pdf_filepath
    )


    return final_state, audit_filepath                  # the file path where the audit trail JSON was actually saved


if __name__ == "__main__":                                                 # this code only runs if you execute main.py directly
    from agents.agent3_report import print_readable_report

    FILE_PATH = os.path.join(BASE_DIR, "data", "CASE-2026-304.pdf")        # builds the full path to a specific test case file, using BASE_DIR (project's root folder) to construct it correctly.
    final_state, audit_filepath = run_pipeline(FILE_PATH)                  # run_pipeline returns two values, and they are unpacked into two separate variables: final_state - gets the first return value and audit_filepath - gets the second return value,   This is called tuple unpacking

    print("\n" + "="*80)
    print("PIPELINE COMPLETE - FINAL REPORT")
    print("="*80)
    print_readable_report(final_state["final_report"])



