from typing import TypedDict, Optional
from schemas import RiskAssessment, ComplianceStatus, FinalReport

# Shared LangGraph state

class IdentityRiskState(TypedDict):
    file_path: str                                    # The specific case PDF to process - set when starting the workflow, read by Agent 1's node
    risk_assessment: Optional[RiskAssessment]         # Agent 1 result
    compliance_status: Optional[ComplianceStatus]     # Agent 2 result
    final_report: Optional[FinalReport]               # Agent 3 result
    audit_log: list                                   # Log of actions taken by each agent