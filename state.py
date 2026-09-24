from typing import TypedDict, Optional
from schemas import RiskAssessment, ComplianceStatus, FinalReport

# Shared LangGraph state

class IdentityRiskState(TypedDict):
    file_path: str                                    # The specific case PDF to process - set when starting the workflow, read by Agent 1's node
    risk_assessment: Optional[RiskAssessment]         # Agent 1 result
    compliance_status: Optional[ComplianceStatus]     # Agent 2 result
    final_report: Optional[FinalReport]               # Agent 3 result
    audit_log: list                                   # Log of actions taken by each agent

    # These following details are already present in the "RiskAssessment", Just for clarification i mentioned here
    '''
    requires_human_review: bool  # Calculated by routing code based on confidence (confidence_score), not the LLM.  {Actually the LLM read the case data and will assign confidence_score (its own self-assessment of certainty),  then routing code will reads that LLM-assigned score, and  will calculates human_review from it using a threshold rule}
    enhanced_due_diligence_required: bool   # Calculated by routing code based on risk/PEP signals, not the LLM  {Calculated by routing code from Agent 1's output: risk_score OR pep_match}

    '''
