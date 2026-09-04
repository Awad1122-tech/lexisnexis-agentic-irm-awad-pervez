from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator # model_validator, the decorator needed to run custom logic before validation


class PreviousAddress(BaseModel):
    address: str = Field(description="The previous address")
    residency_score: str = Field(description="Confidence level the subject lived here, e.g. High, Medium, Low")     # LLM's own assessment of confidence in that address
    sources_count: int = Field(description="Number of data sources confirming this address")
    occupied_period: str = Field(description="Date range this address was occupied, e.g. '2018 - 2026'")
    forwarding_confirmed: Optional[bool] = Field(default=None, description="Whether a forwarding address link was confirmed to the next address, if applicable")
    source_records: List[str] = Field(default_factory=list, description="List of specific source records confirming this address, e.g. 'Registry Trust: Electoral roll entry 2018-2026'")


class Directorship(BaseModel):
    company_name: str = Field(description="Name of the company")
    company_number: str = Field(description="UK Companies House registration number")
    appointed_date: str = Field(description="Date the subject was appointed as director")
    status: Literal["Active", "Dissolved"] = Field(description="Current status of the company")


class PropertyRecord(BaseModel):
    address: str = Field(description="Address of the property")
    owner_confirmed: bool = Field(description="Whether ownership was confirmed via the property register")
    ownership_status: str = Field(description="Detail of ownership, e.g. 'Confirmed current owner since 2019' or 'No register entry - rental property'")
    is_subject_residence: bool = Field(description="True if the subject actually lives here, False if owned but not resided in (e.g. buy-to-let)")


class CCJRecord(BaseModel):
    court: str = Field(description="Court that issued the County Court Judgment")
    date: str = Field(description="Date of the judgment")
    amount: Optional[str] = Field(default=None, description="Judgment amount, if disclosed")


class AdverseMediaArticle(BaseModel):
    description: str = Field(description="Brief description of the adverse media finding")
    evidence_type: Optional[str] = Field(default=None, description="Source of the finding, e.g. 'National news archive search'")


class ScreeningEvidence(BaseModel):
    source_list: str = Field(description="Which specific list/database this match came from, e.g. 'WorldCompliance PEP Database', 'OFAC SDN List'")  # WorldCompliance PEP Database: specific database used to screen for Politically Exposed Persons (PEPs),  # OFAC: Office of Foreign Assets Control, a U.S. Treasury Department agency that enforces economic sanctions against countries, entities, and individuals, # SDN: Specially Designated Nationals, OFAC's actual list of specific people/entities that are sanctioned. 
    match_record: str = Field(description="The specific record details that matched, e.g. name, DOB, category as it appears on the source list")
    match_basis: str = Field(description="Why this was considered a match, e.g. 'Exact name and date of birth match'")



class TraceIQFindings(BaseModel):
    # Identity Tracing (LexID Linking)
    lex_id: Optional[str] = Field(default=None, description="Unique LexID identifier, None if not yet assigned")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/spellings associated with this identity")

    # Current Address Profile
    current_address: str = Field(description="The subject's current address on file")
    residency_score: Literal["Not Calculated", "Very Low", "Low", "Medium", "High", "Very High"] = Field(description="Confidence level the subject genuinely lives at the current address")  # LLM's own assessment
    last_seen: Optional[str] = Field(default=None, description="Most recent date confirming this address")
    sources_count: int = Field(description="Number of data sources backing the current address")
    source_records: List[str] = Field(default_factory=list, description="List of specific source records confirming the current address, e.g. 'Registry Trust: Electoral roll entry confirmed 2019-present'")

    # Property Details
    current_property: Optional[PropertyRecord] = Field(default=None, description="Property record for the subject's current residence")
    additional_properties: List[PropertyRecord] = Field(default_factory=list, description="Any additional owned properties not resided in, e.g. buy-to-let")

    # Occupants
    linked_occupants_count: int = Field(description="Number of other people linked to the current address")
    linked_occupants_names: List[str] = Field(default_factory=list, description="Names of linked occupants, if known")
    occupants_source: Optional[str] = Field(default=None, description="How the occupant links were established, e.g. 'electoral roll cross-match'")

    # Previous Address History
    previous_addresses: List[PreviousAddress] = Field(default_factory=list, description="List of previous addresses on file")

    # Financial and Corporate Records
    mortality_status: Literal["No Match", "Potential Match", "Confirmed Match"] = Field(description="Result of mortality/death register screening")
    ccj_records: List[CCJRecord] = Field(default_factory=list, description="List of County Court Judgments found, each with court, date, and amount")
    has_bankruptcy_iva_dro: bool = Field(description="Whether any bankruptcy, IVA, or Debt Relief Order was found")
    directorships: List[Directorship] = Field(default_factory=list, description="Company directorships associated with this identity")

    """
    # IVA (Individual Voluntary Arrangement): a formal agreement in the UK where someone struggling with debt agrees to pay creditors a set amount over time (usually 5-6 years), 
      avoiding bankruptcy while still settling debts.
    
      
    # DRO ((Debt Relief Order)): a way for someone genuinely too poor to pay their debts to have those debts legally cancelled/written off - specifically for people with very low income, few assets, and 
     relatively small debt (under a set limit). After a 12-month waiting period, if their situation hasn't improved, the debt is officially erased, and they're no longer legally required to pay it.


    # Bankruptcy: a formal legal process where someone who cannot pay their debts declares this to the court, and in exchange, most of their debts are legally written off - but usually at a bigger cost than
      an IVA or DRO (e.g., losing valuable assets like a house or car, and having finances controlled by an official "trustee" for a period, typically 12 months).
    """

    # Linked Associated Individuals
    associated_individuals_note: Optional[str] = Field(default=None, description="Summary note on associated individuals and whether connections are established or unexplained")


    @model_validator(mode='before') # Tells Pydantic: run this function before validating the data against the schema

    @classmethod  # Required by Pydantic - makes this function belong to the class itself, not an instance

    def convert_none_lists_to_empty(cls, data):
        # cls = the TraceIQFindings class itself
        # data = the raw dictionary from the LLM, not yet validated

        if isinstance(data, dict):
            # Only proceed if data is actually a dictionary (normal expected case)

            list_fields = ['aliases', 'additional_properties', 'linked_occupants_names',
                          'previous_addresses', 'ccj_records', 'directorships']
            # The 6 fields that are supposed to be lists, and are at risk of being null

            for field in list_fields:
                # Check each of those 6 fields, one at a time
                # .get(key) is a built-in Python dictionary method
                # that retrieves the value for a given key - safely,
                # without crashing if the key doesn't exist.
                if data.get(field) is None:
                    # If the LLM wrote null for this field

                    data[field] = []
                    # replace it with an empty list instead

        return data
        # Return the corrected data, now safe for Pydantic to validate normally


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    """ Structured output from Agent 1 - Risk Analysis Agent """
    case_id: str = Field(description="Unique case identifier")
    customer_name: str = Field(description="Name of the customer")

    # Threat_Matrix
    risk_score: Optional[float] = Field(default=None, description="Numerical risk score 0-100 from ThreatMetrix, None if not calculable due to insufficient data")  # (Higher = Riskier, and vice versa),  How likely is fraud in this specific transaction right now ?
    trust_score: Optional[float] = Field(default=None, description="Numerical trust score 0-100 from ThreatMetrix, None if not calculable due to insufficient data") # How clean this digital identity looks, lower = more suspicious  (How can I rely on this identity based on its history/ reputation)
    overall_risk_rating: Literal["LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"] = Field(description="Overall risk rating") # overall_risk_rating in Schema is not something extracted from raw text. it's something Agent 1's LLM must calculate/determine itself, based on reading the raw signals (Trust Score, Risk Score, PEP match, mortality status, CCJs, etc.) and forming its own judgment.
    key_signals: List[str] = Field(description="List of key risk signals identified across all products")  # key_signals is more extraction pulling out flags that are already explicitly stated in the raw data and compiling them into a clean list


    # Identity verification (IDU + TrueID)
    identity_verification_result: Literal["PASS", "FAIL", "INSUFFICIENT", "PASS (marginal)"] = Field(description="Overall result from IDU / Instant Verify International")
    document_type: str = Field(description="Type of identity document submitted, e.g. 'UK Passport', 'International Passport', 'EU National ID'")
    document_authentication_result: Literal["Pass", "Fail"] = Field(description="Pass/Fail result of document authentication from TrueID")
    document_authentication_score: float = Field(description="Document authentication score 0-100 from TrueID") # high score, document is genuine, not forged. Low score = signs of tampering/forgery
    facial_match_score: float = Field(description="Facial biometric match score 0-100 from TrueID") # Does the selfie match the document photo, high = the selfie genuinely matches the document photo, Low = doesn't convincingly match - could be a different person
    liveness_check_passed: bool = Field(description="Whether the liveness check passed") # Is this a real live person, not a photo of a photo

    # PEP and Sanctions screening (Bridger Insight XG)
    pep_match: bool = Field(description="True if a Politically Exposed Person (PEP) match was identified for this case, False otherwise")
    pep_confidence: Optional[float] = Field(default=None, description="Confidence score 0-100 for PEP match, if applicable") # High confidence (e.g., 92%) → very likely this really is a PEP/sanctioned person -> treat seriously,   Low confidence (e.g., 58%) -> could genuinely be them, or could be a coincidental name match -> needs more verification before acting on it [LLM-derived, but grounded in real data.]
    pep_evidence: Optional[ScreeningEvidence] = Field(default=None, description="Source list, matched record, and match basis for the PEP match, if applicable")
    sanctions_match: bool = Field(description="True if a sanctions list match was identified, False otherwise")
    sanctions_confidence: Optional[float] = Field(default=None, description="Confidence score 0-100 for sanctions match, if applicable")  # LLM-derived, grounded in stated evidence (e.g., "exact match" vs. "close name match, transliteration variant")
    sanctions_evidence: Optional[ScreeningEvidence] = Field(default=None, description="Source list, matched record, and match basis for the sanctions match, if applicable")
    adverse_media_count: int = Field(description="Total number of adverse media articles found")
    adverse_media_articles: List[AdverseMediaArticle] = Field(default_factory=list, description="List of adverse media articles found, each with a description and evidence source")
    state_owned_enterprise_flag: bool = Field(description="Whether the subject is linked to a state-owned enterprise")

    # Due diligence and human review flags
    confidence_score: float = Field(description="Overall confidence score 0-100 in this risk assessment") # LLM extracts from data. The LLM's own stated certainty in its assessment (this is a field the LLM fills in itself, not calculated by code, and not literally printed in the case PDF, it's the LLM's self-assessment of how sure it is, basically the LLM saying "how sure am I that my analysis of this case is correct?
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Confidence level of assessment") # LLM extracts/derives. Same as above - LLM's own label (LOW/MEDIUM/HIGH) for its confidence, alongside the numeric version.
    requires_human_review: bool = Field(description="Whether human review is required") # CODE calculates. Not from the LLM, not from the raw data - routing function computes this (based on confidence_score)
    enhanced_due_diligence_required: bool = Field(description="Whether enhanced due diligence is required for this case") # CODE calculates. Routing function computes this (based on risk_score and pep_match)
    enhanced_due_diligence_completed: Optional[bool] = Field(default=None, description="Whether EDD was completed - None if not yet tracked by this system, requires human analyst confirmation")   # From terminal
    # TraceIQ integration + audit fields
    trace_iq: TraceIQFindings = Field(description="Structured findings from TraceIQ identity tracing")  # LLM reads the raw TraceIQ section of the case text and extracts/structures it into the TraceIQFindings object
    detailed_analysis: str = Field(description="Full detailed explanation of each risk signal identified, written as connected paragraphs") # The LLM writes original explanatory prose connecting all the risk signals together
    recommended_action: str = Field(description="Recommended action for the analyst")  # the LLM makes a judgment call, recommending an action based on its analysis of the case
    next_steps: List[str] = Field(description="Suggested next steps") # LLM generates its own suggested next steps based on its analysis, not something extracted from the raw case text
    regulatory_considerations: List[str] = Field(description="Relevant regulatory considerations (ungrounded, LLM's own assessment)") # Ungrounded = LLM's own guess from memory, not backed by retrieved documents


    @model_validator(mode='before')
    @classmethod
    def convert_none_lists_to_empty(cls, data):
        if isinstance(data, dict):
            list_fields = ['key_signals', 'adverse_media_articles', 'next_steps', 'regulatory_considerations']
            for field in list_fields:
                if data.get(field) is None:
                    data[field] = []
            return data



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ComplianceStatus(BaseModel):

    # core compliance verdicts

    """ Structured output from agent 2 - Compliance Agent """
    case_id: str = Field(description="Unique case identifier")
    gdpr_compliant: bool = Field(description="whether case meets GDPR requirements")
    fca_compliant: bool = Field(description="whether case meets FCA requirements")
    aml_kyc_compliant: Literal["Compliant", "Non-Compliant", "Pending Verification"] = Field(description="AML/KYC compliance status - Pending Verification when EDD completion status is not yet confirmed")

    mortality_confirmed_block: bool = Field(description="True if a CONFIRMED mortality match was found (subject is deceased). When True, this forces all compliance fields to False and the case must be automatically rejected, regardless of any other findings.")
    # If the real person is confirmed dead, whoever is applying is definitely not them - this is fraud.
    # Once this is true, we don't need to check anything else (GDPR, FCA, AML) - the case fails immediately,
    # no matter how clean everything else looks.


    compliance_flags: List[str] = Field(description="List of specific compliance problems found, in plain language")
    '''Agent 2's RAG pipeline retrieves real text from GDPR.pdf, FCA.pdf, PEP.pdf, etc. based on this case's signals. The LLM reads that retrieved text + Agent 1's case facts, then writes each
       specific problem it finds as one plain sentence here - e.g. "PEP match without confirmed EDD (Reg. 35, MLR 2017)". This is grounded in the actual retrieved regulation, not just LLM memory.
    '''

    regulatory_actions_required: List[str] = Field(description="List of specific actions required to bring this case into compliance")
    '''Same retrieved regulatory text is used here, but instead of describing WHAT is wrong (that's compliance_flags), this field describes WHAT TO DO about it - the specific fix, based on what
       the actual retrieved regulation requires - e.g. "Complete enhanced due diligence for PEP match"
    '''

class FCAFinalCheck(BaseModel):
    """
    A SEPARATE, final FCA verdict - produced AFTER Agent 3 finishes, using the
    complete decision_rationale and full audit trail (which don't exist yet at
    Agent 2's point in the pipeline). This is more complete than Agent 2's
    early-stage FCA check, which can only assess Agent 1's detailed_analysis.
    
    """
    fca_compliant_final: bool = Field(description="Final FCA compliance verdict, based on the complete decision rationale and audit trail")
    reason: str = Field(description="Brief explanation citing specific evidence from the decision rationale or audit trail")


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ReportSynthesis(BaseModel):
    """
    Only the fields Agent 3's LLM actually needs to generate - everything else in FinalReport is already known and filled in afterward, not regenerated by the LLM

    """

    executive_summary: str = Field(description="Brief executive summary")
    overall_risk_rating: str = Field(description="Final overall risk rating")
    decision_rationale: str = Field(description="A clear, human-readable explanation of WHY this overall decision was reached, connecting Agent 1's risk signals, Agent 2's compliance findings, and the resulting recommendation into one coherent narrative")
    final_recommendations: List[str] = Field(description="Final recommendations")
    final_next_steps: List[str] = Field(description="Final suggested next steps")


class FinalReport(BaseModel):
    """ The complete final report - built by combining Agent 3's LLM synthesis with data already known from Agent 1 and Agent 2 """

    case_id: str = Field(description="Unique case identifier")
    customer_name: str = Field(description="Name of the customer")

    executive_summary: str = Field(description="Brief executive summary")
    overall_risk_rating: str = Field(description="Final overall risk rating")
    decision_rationale: str = Field(description="Explanation of WHY this decision was reached")
    final_recommendations: List[str] = Field(description="Final recommendations")
    final_next_steps: List[str] = Field(description="Final suggested next steps")

    risk_assessment: RiskAssessment = Field(description="Full risk assessment from Agent 1")
    compliance_status: ComplianceStatus = Field(description="Full compliance status from Agent 2")
    audit_trail_reference: str = Field(description="Reference to audit trail file")
    report_timestamp: str = Field(description="Timestamp of report generation")