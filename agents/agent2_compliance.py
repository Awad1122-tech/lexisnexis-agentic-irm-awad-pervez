### Agent 2: Compliance Agent

"""
RAG PIPELINE OVER REGULATORY DOCUMENTS (GDPR, FCA, AML/KYC: CDD/EDD/PEP).
Takes Agents 1's RiskAssessment and produces structured ComplianceStatus.
Single entry point: run_agent2(risk_assessment) -> ComplianceStatus

"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from schemas import ComplianceStatus, RiskAssessment, FCAFinalCheck
import os


load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGULATORY_DOCS_PATH = os.path.join(BASE_DIR, 'knowledge_base', 'regulatory_docs')


##  Step 1 of RAG Pipeline: (Document Ingestion) Load ALL regulatory PDFs from the folder using DirectoryLoader

def load_regulatory_docs():
    loader = PyPDFDirectoryLoader(REGULATORY_DOCS_PATH)   # PyPDFDirectoryLoader, load all files from the folder
    docs = loader.load()
    return docs              # docs = list of Document objects (one per PDF page), each with: page_content, and metadata



## Step 2 of RAG Pipeline (Text Chunking): Split regulatory documents into smaller chunks

# I am using the combination of chunk_size = 800 and chunk_overlap = 100, because of the text-dense documents

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100    
    )
    chunks = splitter.split_documents(docs)
    return chunks


# vector store creation (Embeddings and ChromaDB)

def create_regulatory_vector_store(chunks):
    embedding_model = HuggingFaceEndpointEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2',
        task='feature-extraction'
    )

    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=os.path.join(BASE_DIR, 'chroma_db_regulatory'),
        collection_name='regulatory_docs'
    )

    """
    Working Logic: vector_store.get() retrieves everything currently stored in this collection if it's the very first time running Agent 2, 
    the collection is empty, so existing['ids'] is an empty list [], and len() == 0 is True → embeds the documents.
    On every subsequent run, the collection already has data, existing['ids'] has entries, len() > 0 → skips re-embedding.
    """

    existing = vector_store.get()
    if len(existing['ids']) == 0:

        vector_store.add_documents(chunks)  # vector_store.add_documents(chunks) = embed (using embedding_model) + store (into ChromaDB), both happening together, in that exact order, within this single line.

    return vector_store


# Creating retriever to search ChromaDB based on similarity

def create_retriever(vector_store):
    
    retriever = vector_store.as_retriever(   #.as_retriever() is a method that converts your vector_store (the Chroma database object) into a proper Retriever object - which is a Runnable, meaning it can be used with .invoke() and chained with |.
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    return retriever



"""
    Unlike Agent 1, which runs ONE broad search (k=12) covering an entire case, document spanning many different topics (ThreatMetrix, IDU, TrueID, TraceIQ, Bridger all mixed together),
    Agent 2 will run THREE SEPARATE searches, each narrowly focused on just ONE regulatory topic:
        1. GDPR automated decision-making safeguards
        2. FCA accountability and explainability
        3. AML/KYC - CDD, EDD, PEP requirements

    Because each individual search only needs to find chunks relevant to ONE focused topic (not many topics mixed together), it needs fewer chunks per search to get good coverage 
    hence k=6 here, versus k=12 for Agent 1's single all-encompassing search.

    Note: total chunks retrieved across Agent 2's 3 searches (6+6+6=18) is actually MORE than Agent 1's single search (12) - just split across  3 precise, targeted queries instead of one broad one.
"""


# Augmentation 



def create_prompt():
    parser = PydanticOutputParser(pydantic_object=ComplianceStatus)  # Create a parser tied to the ComplianceStatus schema
    format_instructions = parser.get_format_instructions()

    # ChatPromptTemplate is LangChain's tool for building a reusable prompt with placeholders (like {context}, {question}) that get filled in with real values later, when the chain actually runs.

    compliance_prompt = ChatPromptTemplate.from_messages([                                            # from_messages:  role-based messages
        ("system", """You are an expert compliance analyst specialising in UK financial
         services regulation. You assess identity risk cases against GDPR (Data Use and
         Access Act 2025, Articles 22A-22D), FCA principles (AI Update 2024), and AML/KYC
         requirements (Money Laundering Regulations 2017 - Customer Due Diligence,
         Enhanced Due Diligence, and PEP screening).
         Base every compliance judgement strictly on the retrieved regulatory text provided.
         Never rely on general knowledge - only the specific regulatory context given."""),

        ("human", """Based on the following retrieved regulatory text and the risk
         assessment findings, determine this case's compliance status.

        Retrieved Regulatory Context:
        {context}

        Case Risk Assessment Summary:
        {risk_summary}

        Apply EXACTLY the following rules, in order. Each rule's condition must be
        checked precisely against the values given above - do not infer, generalise,
        or apply a rule based on related-but-different values.

        IMPORTANT: You MUST produce compliance_flags entries covering ALL THREE
        compliance areas - GDPR, FCA, AND AML/KYC - every time, regardless of outcome.
        Do not omit any area, even when it is fully compliant with no issues. For each
        of the three areas, state clearly whether it is compliant or not, and why,
        citing the specific values checked. This applies even if only 1-2 areas have
        actual problems - the compliant area(s) still require their own explicit
        compliance_flags entry, not just their own regulatory_actions_required entry.
        Similarly, produce a regulatory_actions_required entry for ALL THREE areas -
        stating the specific action needed if non-compliant, or "No action required"
        if compliant. Each regulatory_actions_required entry MUST begin by naming
        which compliance area it addresses (e.g. "GDPR:", "FCA:", "AML/KYC:"), even
        when the entry is "No action required" - do not write a bare "No action
        required" without specifying which area it refers to.

        RULE 1 (Mortality - Confirmed): IF Mortality Status is exactly "Confirmed Match":
            -> Set gdpr_compliant to False, fca_compliant to False, AND
               aml_kyc_compliant to "Non-Compliant".
            -> This overrides every other rule below. Stop here if this rule applies.

        RULE 2 (Mortality - Potential): IF Mortality Status is exactly "Potential Match"
        (and Rule 1 did NOT apply):
            -> Set aml_kyc_compliant to EXACTLY "Non-Compliant" - NOT "Pending
               Verification". "Pending Verification" is reserved specifically for
               Rule 6's EDD-completion-unknown scenario and must not be used here,
               even though both situations involve genuine uncertainty requiring
               manual verification.
            -> Add a compliance flag noting manual verification of mortality status is
               required.
            -> Do NOT set gdpr_compliant or fca_compliant to False based on this alone.
            -> This rule takes priority over Rule 6 for determining aml_kyc_compliant -
               if Rule 2 applies, use its "Non-Compliant" outcome even if Rule 6 would
               otherwise suggest "Pending Verification".

        RULE 3 (GDPR - Automated Decision Safeguards): Check the EXACT value of
        "Requires Human Review" above.
            -> IF Requires Human Review is exactly False: set gdpr_compliant to False
               (no human review means Article 22C safeguards are required but not
               currently implemented in this system).
            -> IF Requires Human Review is exactly True: set gdpr_compliant to True
               (a human is reviewing this case, so the automated-decision safeguard
               concern does not apply). Do NOT set gdpr_compliant to False in this case.
            -> REQUIRED: produce exactly TWO SEPARATE entries in compliance_flags
               for this rule (not one combined sentence):
               Entry 1: state only whether a human is reviewing this case - e.g.
               "No human is reviewing this case." or "A human is reviewing this
               case."
               Entry 2 MUST be EXACTLY this sentence, with only the number filled
               in: "Confidence Score of ___ compared against the threshold of 70."
               Replace ___ with the actual Confidence Score value from this case.
               Do not add, remove, or rephrase any other word in this sentence.
            -> REQUIRED: the regulatory_actions_required entry for this rule must
               name "Article 22C" followed immediately by its definition in
               parentheses, in this exact form: 'Article 22C (which requires
               [specific requirement])'. This must read as a clear definition, not
               an explanation woven into a longer sentence.
               

        RULE 4 (FCA - Accountability and Explainability): Base your decision on the
        RETRIEVED FCA REGULATORY CONTEXT above (not general knowledge) - identify what
        it requires regarding accountability and explainability, then assess whether
        the Detailed Analysis satisfies that requirement.
            -> Set fca_compliant to True or False.
            -> REQUIRED: add one entry to compliance_flags that clearly explains your
               decision, referencing the specific FCA principle from the retrieved
               context that you applied.
            -> Do NOT cite human review status as the reason for fca_compliant - human
               review is Rule 3's (GDPR's) concern. Rule 4's decision must be based on
               whether the Detailed Analysis itself is substantive, not on whether a
               human is reviewing the case.

        RULE 5 (AML/KYC - Customer Due Diligence, Regulation 28): Check the EXACT
        values of "Identity Verification Result (IDU)" and "Document Authentication
        Result (TrueID)" above.
            -> IF Identity Verification Result is exactly "FAIL" OR "INSUFFICIENT", OR
               Document Authentication Result is exactly "Fail": set aml_kyc_compliant
               to "Non-Compliant" (basic customer due diligence could not be
               satisfactorily completed).
            -> IF both are "PASS"/"Pass" (in any form, including "PASS (marginal)"):
               this rule does not apply - proceed to Rule 6 to determine the final
               aml_kyc_compliant value.

        MANDATORY CHECK before finalizing aml_kyc_compliant: re-verify the EXACT
        value of "Document Authentication Result" above. If it is exactly "Fail",
        aml_kyc_compliant MUST be "Non-Compliant" - this is not optional and
        cannot be overridden by any other consideration. Setting aml_kyc_compliant
        to "Compliant" when Document Authentication Result is "Fail" is an error.

        Note: Rule 5 has ALREADY been evaluated by the time you reach Rule 6. If Rule 5
        set aml_kyc_compliant to "Non-Compliant", that value is FINAL and Rule 6 must
        NOT change it to "Compliant" under any branch, including OTHERWISE.      

        Note: Before writing any compliance_flags entry mentioning EDD, verify the
        EXACT value of "Enhanced Due Diligence Required" above. Do NOT state "EDD is
        required" unless this field is literally True - even if the case seems
        intuitively to warrant closer scrutiny (e.g. an INSUFFICIENT identity
        verification result), the EDD requirement is determined ONLY by this field's
        actual value, not by general risk intuition.

        RULE 6 (AML/KYC - Enhanced Due Diligence, Regulations 33 and 35): Check the
        EXACT values of "PEP Match", "Enhanced Due Diligence Required", and "Enhanced
        Due Diligence Completed" above - do NOT infer one field from another.

            BRANCH A1 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            explicitly False: set aml_kyc_compliant to "Non-Compliant".

            BRANCH A2 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            None: set aml_kyc_compliant to "Pending Verification". Add compliance flag:
            "PEP match confirmed; EDD completion not yet verified - pending manual
            confirmation."

            BRANCH A3 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            explicitly True: this branch does NOT independently fail
            aml_kyc_compliant. If Rule 5 already set aml_kyc_compliant to
            "Non-Compliant" earlier, that outcome stands - do not overwrite it. If
            Rule 5 did NOT apply, set aml_kyc_compliant to "Compliant" here.

            BRANCH B1 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is explicitly False: set
            aml_kyc_compliant to "Non-Compliant".

            BRANCH B2 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is None: set aml_kyc_compliant to
            "Pending Verification". Add compliance flag: "Enhanced due diligence
            required; completion not yet verified - pending manual confirmation."

            BRANCH B3 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is explicitly True: this branch
            does NOT independently fail aml_kyc_compliant. If Rule 5 already set
            aml_kyc_compliant to "Non-Compliant" earlier, that outcome stands - do not
            overwrite it. If Rule 5 did NOT apply, set aml_kyc_compliant to
            "Compliant" here.

            OTHERWISE (Enhanced Due Diligence Required is False, OR Enhanced Due
            Diligence Completed is explicitly True per Branch A3/B3 above, or no
            PEP/EDD concern applies): set aml_kyc_compliant to "Compliant", UNLESS
            Rule 5 already set it to "Non-Compliant" earlier - in that case, that
            outcome stands.

        After applying all rules above:
        7. List every compliance flag your rules triggered, in plain language,
           grounded in the retrieved regulatory text and the specific values checked.
        8. List specific regulatory actions required to bring the case into full
           compliance, one action per flag raised.

        {format_instructions}""")
    ])

    return compliance_prompt.partial(format_instructions=format_instructions)


def create_fca_final_check_prompt():
    parser = PydanticOutputParser(pydantic_object=FCAFinalCheck)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert compliance reviewer performing a FINAL FCA
         accountability and explainability check, using the complete, finished
         decision rationale and full audit trail - the most complete evidence
         available at any point in this system's pipeline."""),

        ("human", """Based on the following RETRIEVED FCA REGULATORY TEXT, detailed
         analysis, decision rationale, and audit trail, assess whether this case
         demonstrates genuine FCA accountability and explainability - grounded in
         the actual retrieved text, not general knowledge.

        Retrieved FCA Regulatory Context:
        {fca_context} 

        Detailed Analysis:
        {detailed_analysis}

        Decision Rationale:
        {decision_rationale}

        Audit Trail Summary:
        {audit_summary}

        1. FIRST, read the RETRIEVED FCA REGULATORY CONTEXT above and identify
           the specific requirement(s) it sets out (e.g. what Consumer Duty or
           SM&CR actually requires regarding accountability and explainability).
           THEN, check whether the Detailed Analysis, Decision Rationale, and
           Audit Trail Summary - TOGETHER - actually satisfy that specific
           requirement. Set fca_compliant_final to True only if they genuinely
           meet the standard set by the retrieved text; set it to False if they
           fall short of what the retrieved text actually requires.
        2. CRITICAL: FCA compliance is INDEPENDENT of GDPR and AML/KYC compliance
           status. Do NOT set fca_compliant_final to False merely because the
           decision rationale mentions that GDPR or AML/KYC is non-compliant -
           those are SEPARATE regulatory areas with their own rules. Specifically,
           the ABSENCE of human review (which triggers GDPR Article 22C concerns)
           is NOT, by itself, evidence of an FCA/SM&CR accountability failure -
           do not cite "no human is reviewing this case" or "lack of human
           review" as a reason for fca_compliant_final=False. FCA's
           accountability/explainability requirement is about the QUALITY of
           the reasoning and record-keeping itself, not whether a human
           reviewed the case. A case can have excellent FCA accountability (a
           clear, well-reasoned rationale, complete audit trail) even while
           having no human review and being GDPR non-compliant - these are not
           linked. Similarly, do NOT set fca_compliant_final to True simply
           because GDPR or AML/KYC happen to be compliant - FCA's verdict must
           be based solely on the retrieved FCA text and the quality of the
           rationale/audit trail, regardless of what the other 2 areas' status
           happens to be, in either direction.
        3. Provide a brief reason for this verdict, referencing the specific FCA
           principle from the retrieved context AND citing evidence from the
           Detailed Analysis, Decision Rationale, or Audit Trail Summary - NOT
           citing GDPR or AML/KYC status as the reason.
        {format_instructions}""")
    ])

    return prompt.partial(format_instructions=format_instructions)


def run_fca_final_check(retriever, detailed_analysis: str, decision_rationale: str, audit_log: list) -> FCAFinalCheck:

    fca_docs = retriever.invoke("FCA Consumer Duty SM&CR accountability explainability")
    fca_context = "\n\n".join(doc.page_content for doc in fca_docs)

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        max_new_tokens=1000
    )
    model = ChatHuggingFace(llm=llm)

    parser = PydanticOutputParser(pydantic_object=FCAFinalCheck)
    prompt = create_fca_final_check_prompt()

    audit_summary = "\n".join(
        f"[{entry['timestamp']}] {entry['step']}: {entry['action']}"
        for entry in audit_log
    )

    chain = prompt | model | parser

    result = chain.invoke({
        "fca_context": fca_context,
        "detailed_analysis": detailed_analysis,
        "decision_rationale": decision_rationale,
        "audit_summary": audit_summary
    })

    return result


# Generation (Generating Response using LLM) using chains

def run_compliance_analysis(retriever, risk_assessment: RiskAssessment):

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    max_new_tokens=2000
    )

    model = ChatHuggingFace(llm=llm)


    parser = PydanticOutputParser(pydantic_object=ComplianceStatus)
    compliance_prompt = create_prompt()


    gdpr_query = "GDPR automated decision-making safeguards Article 22 human intervention right to contest"
    fca_query = "FCA accountability explainability fairness Consumer Duty Senior Managers Certification Regime"
    aml_query = "AML KYC customer due diligence enhanced due diligence PEP politically exposed person"


    gdpr_docs = retriever.invoke(gdpr_query)
    fca_docs = retriever.invoke(fca_query)
    aml_docs = retriever.invoke(aml_query)


    context = "\n\n".join(doc.page_content for doc in gdpr_docs + fca_docs + aml_docs)


    risk_summary = f"""
    Case ID: {risk_assessment.case_id}
    Requires Human Review: {risk_assessment.requires_human_review}
    PEP Match: {risk_assessment.pep_match}
    Enhanced Due Diligence Required: {risk_assessment.enhanced_due_diligence_required}
    Enhanced Due Diligence Completed: {risk_assessment.enhanced_due_diligence_completed}
    Mortality Status: {risk_assessment.trace_iq.mortality_status}
    Overall Risk Rating: {risk_assessment.overall_risk_rating}
    Confidence Score: {risk_assessment.confidence_score}
    Identity Verification Result: {risk_assessment.identity_verification_result}
    Document Authentication Result: {risk_assessment.document_authentication_result}
    Detailed Analysis: {risk_assessment.detailed_analysis}
    """

    chain = compliance_prompt | model | parser

    result = chain.invoke({
        "context": context,
        "risk_summary": risk_summary
    })

    return result


def run_agent2(risk_assessment: RiskAssessment) -> ComplianceStatus:
    docs = load_regulatory_docs()
    chunks = split_documents(docs)
    vector_store = create_regulatory_vector_store(chunks)
    retriever = create_retriever(vector_store)

    compliance_status = run_compliance_analysis(retriever, risk_assessment)
    compliance_status.case_id = risk_assessment.case_id  # overwrite with reliable case_id from Agent 1

    return compliance_status

'''

if __name__ == "__main__":
    from agents.agent1_risk import run_agent1

    FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-301.pdf')
    risk_assessment = run_agent1(FILE_PATH)
    compliance_status = run_agent2(risk_assessment)
    print(compliance_status.case_id)
    print(compliance_status)

'''

if __name__ == "__main__":
    from agents.agent1_risk import run_agent1

    FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-316.pdf')
    risk_assessment = run_agent1(FILE_PATH)

    print("\n\n" + "="*80)
    print("AGENT 1 OUTPUT (relevant fields only, used as input for Agent 2)")
    print("="*80)
    print(f"Requires Human Review: {risk_assessment.requires_human_review}")
    print(f"Detailed Analysis: {risk_assessment.detailed_analysis}")
    print(f"Identity Verification Result (IDU): {risk_assessment.identity_verification_result}")
    print(f"Document Authentication Result (TrueID): {risk_assessment.document_authentication_result}")
    print(f"Enhanced Due Diligence Required: {risk_assessment.enhanced_due_diligence_required}")
    print(f"Enhanced Due Diligence Completed: {risk_assessment.enhanced_due_diligence_completed}")
    print(f"PEP Match: {risk_assessment.pep_match}")
    print(f"Mortality Status: {risk_assessment.trace_iq.mortality_status}")

    compliance_status = run_agent2(risk_assessment)

    print("\n\n" + "="*80)
    print("AGENT 2 OUTPUT (compliance verdict)")
    print("="*80)
    print(compliance_status)
    print("\n" + "="*80 + "\n")



# TODO: enhanced_due_diligence_completed - future implementation plan
#
# WHAT: This field tracks whether a human analyst has actually carried out
# Enhanced Due Diligence (source of wealth check, senior management approval,
# enhanced monitoring setup) for a specific case - a real-world action, not
# something extractable from any identity risk report.
#
# WHERE IT BELONGS: main.py's workflow/state layer - NOT Agent 1, NOT Agent 2.
# Neither agent's LLM should generate this value, since it's not a fact about
# the applicant or the case data - it's a fact about what YOUR OWN COMPANY'S
# compliance team has actually done, tracked by your system's workflow, not
# read from text.
#
# HOW IT WOULD WORK, once built:
# 1. Case runs through Agent 1 -> Agent 2 -> Agent 3, producing a FinalReport.
# 2. If enhanced_due_diligence_required is True, main.py flags the case for
#    a human analyst (e.g. via the notification/reviewer system discussed
#    earlier), rather than assuming completion or non-completion.
# 3. The human analyst performs the real EDD work (verifies source of wealth,
#    gets senior sign-off, sets up monitoring).
# 4. The analyst then manually confirms completion in the system - e.g. a
#    simple CLI prompt, a database flag, or a field in the notification
#    reply-handling logic already planned.
# 5. main.py stores this confirmed value (True or False) in state, and Agent 2
#    uses the REAL confirmed value instead of the current default of None.
#
# CURRENT STATE: defaults to None (not yet confirmed) everywhere, since no
# tracking mechanism exists yet. Agent 2's compliance rules treat None as
# "Pending Verification" rather than a hard failure - this is honest and
# correct until step 1-5 above are built.



# TODO: aml_kyc_compliant is now a 3-state field (Compliant / Non-Compliant /
# Pending Verification) instead of a plain bool, specifically to handle the
# enhanced_due_diligence_completed=None case honestly - a "Pending Verification"
# result reflects genuine uncertainty (EDD required, but completion not yet
# confirmed by a human analyst) rather than falsely asserting compliance (True)
# or falsely asserting failure (False). Once the human-confirmation feature
# above is built, "Pending Verification" cases should be automatically
# re-evaluated once enhanced_due_diligence_completed changes from None to a
# real True/False value.