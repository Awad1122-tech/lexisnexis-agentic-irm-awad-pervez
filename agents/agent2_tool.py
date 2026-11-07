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
from schemas import ComplianceStatus, RiskAssessment
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



#-------------------------------------------------------------   tool  -------------------------------------------------------------------------------------

def create_search_tool(retriever):
    @tool
    def search_regulations(query: str) -> str:
        """Search the regulatory knowledge base (GDPR, FCA, AML/KYC - CDD, EDD, PEP)
        for text relevant to a specific compliance question. Call this multiple
        times with different queries if you need to check multiple regulatory areas."""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return search_regulations


#----------------------------------------------------------------------------------------------------------------------------------------------------------------


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

        RULE 4 (FCA - Accountability and Explainability): Base your decision on the
        RETRIEVED FCA REGULATORY CONTEXT above (not general knowledge) - identify what
        it requires regarding accountability and explainability, then assess whether
        the Detailed Analysis satisfies that requirement.
            -> Set fca_compliant to True or False.
            -> REQUIRED: add one entry to compliance_flags that clearly explains your
               decision, referencing the specific FCA principle from the retrieved
               context that you applied.

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

        RULE 6 (AML/KYC - Enhanced Due Diligence, Regulations 33 and 35): Check the
        EXACT values of "PEP Match", "Enhanced Due Diligence Required", and "Enhanced
        Due Diligence Completed" above - do NOT infer one field from another.

            BRANCH A1 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            explicitly False: set aml_kyc_compliant to "Non-Compliant".

            BRANCH A2 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            None: set aml_kyc_compliant to "Pending Verification". Add compliance flag:
            "PEP match confirmed; EDD completion not yet verified - pending manual
            confirmation."

            BRANCH B1 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is explicitly False: set
            aml_kyc_compliant to "Non-Compliant".

            BRANCH B2 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is None: set aml_kyc_compliant to
            "Pending Verification". Add compliance flag: "Enhanced due diligence
            required; completion not yet verified - pending manual confirmation."

            OTHERWISE (Enhanced Due Diligence Required is False, or no PEP/EDD concern
            applies): set aml_kyc_compliant to "Compliant", unless Rule 5 (identity/
            document verification failure) independently applies, in which case follow
            Rule 5's outcome instead.

        After applying all rules above:
        7. List every compliance flag your rules triggered, in plain language,
           grounded in the retrieved regulatory text and the specific values checked.
        8. List specific regulatory actions required to bring the case into full
           compliance, one action per flag raised.

        {format_instructions}""")
    ])

    return compliance_prompt.partial(format_instructions=format_instructions)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_tool_calling_prompt():
    parser = PydanticOutputParser(pydantic_object=ComplianceStatus)  # Create a parser tied to the ComplianceStatus schema
    format_instructions = parser.get_format_instructions()

    compliance_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert compliance analyst specialising in UK financial
         services regulation. You assess identity risk cases against GDPR (Data Use and
         Access Act 2025, Articles 22A-22D), FCA principles (AI Update 2024), and AML/KYC
         requirements (Money Laundering Regulations 2017 - Customer Due Diligence,
         Enhanced Due Diligence, and PEP screening).

         You have access to a search_regulations tool that searches the actual
         regulatory documents. Use it to find the specific regulatory text relevant
         to THIS case before making any compliance judgement - search for whatever
         topics are actually relevant given the case facts below, and search multiple
         times if needed to cover different regulatory areas.
         Base every compliance judgement strictly on the regulatory text you retrieve
         using this tool. Never rely on general knowledge - only what you actually
         retrieve."""),

        ("human", """Case Risk Assessment Summary:
        {risk_summary}

        Search the regulatory knowledge base as needed for the rules below, then
        apply EXACTLY the following rules, in order. Each rule's condition must be
        checked precisely against the values given above - do not infer, generalise,
        or apply a rule based on related-but-different values.

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

        RULE 4 (FCA - Accountability and Explainability): Search the regulatory
        knowledge base for FCA principles relevant to THIS specific case's risk
        profile and accountability/explainability requirements - formulate your own
        search query based on what this case actually needs, not a generic template.
        Base your decision on what you retrieve (not general knowledge) - identify
        what it requires, then assess whether the Detailed Analysis satisfies that
        requirement.
            -> Set fca_compliant to True or False.
            -> REQUIRED: add AT LEAST one entry to compliance_flags that clearly
               explains your decision, referencing the specific FCA principle from
               what you retrieved that you applied. This is IN ADDITION to any flags
               required by other rules (e.g. mortality, EDD) - never omit or replace
               another rule's required flag with this one. Your explanation MUST
               logically match your fca_compliant value: if fca_compliant is True,
               your explanation must describe why requirements ARE met; if
               fca_compliant is False, your explanation must describe why
               requirements are NOT met. Do not set fca_compliant to True while
               describing reasons the requirements are not satisfied, or vice versa.

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

        RULE 6 (AML/KYC - Enhanced Due Diligence, Regulations 33 and 35): Check the
        EXACT values of "PEP Match", "Enhanced Due Diligence Required", and "Enhanced
        Due Diligence Completed" above - do NOT infer one field from another.

            BRANCH A1 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            explicitly False: set aml_kyc_compliant to "Non-Compliant".

            BRANCH A2 - IF PEP Match is True AND Enhanced Due Diligence Completed is
            None: set aml_kyc_compliant to "Pending Verification". Add compliance flag:
            "PEP match confirmed; EDD completion not yet verified - pending manual
            confirmation."

            BRANCH B1 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is explicitly False: set
            aml_kyc_compliant to "Non-Compliant".

            BRANCH B2 - IF PEP Match is False AND Enhanced Due Diligence Required is
            True AND Enhanced Due Diligence Completed is None: set aml_kyc_compliant to
            "Pending Verification". Add compliance flag: "Enhanced due diligence
            required; completion not yet verified - pending manual confirmation."

            OTHERWISE (Enhanced Due Diligence Required is False, or no PEP/EDD concern
            applies): set aml_kyc_compliant to "Compliant", unless Rule 5 (identity/
            document verification failure) independently applies, in which case follow
            Rule 5's outcome instead.

        After applying all rules above:
        7. List every compliance flag your rules triggered, in plain language,
           grounded in the regulatory text you retrieved and the specific values checked.
        8. List specific regulatory actions required to bring the case into full
           compliance, one action per flag raised.

        {format_instructions}""")
    ])

    return compliance_prompt.partial(format_instructions=format_instructions)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Generation (Generating Response using LLM) using chains

def run_compliance_analysis(retriever, risk_assessment: RiskAssessment):

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    max_new_tokens=3000
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


# --------------------------------------------------------------------------------------------------------------------------------------------

def run_compliance_analysis_tool_calling(retriever, risk_assessment: RiskAssessment):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        max_new_tokens=3000
    )
    model = ChatHuggingFace(llm=llm)

    search_tool = create_search_tool(retriever)
    model_with_tools = model.bind_tools([search_tool])

    compliance_prompt = create_tool_calling_prompt()

    risk_summary = f"""
    Case ID: {risk_assessment.case_id}
    Requires Human Review: {risk_assessment.requires_human_review}
    PEP Match: {risk_assessment.pep_match}
    Enhanced Due Diligence Required: {risk_assessment.enhanced_due_diligence_required}
    Enhanced Due Diligence Completed: {risk_assessment.enhanced_due_diligence_completed}
    Mortality Status: {risk_assessment.trace_iq.mortality_status}
    Overall Risk Rating: {risk_assessment.overall_risk_rating}
    Identity Verification Result: {risk_assessment.identity_verification_result}
    Document Authentication Result: {risk_assessment.document_authentication_result}
    Detailed Analysis: {risk_assessment.detailed_analysis}
    """


    # messages is a list of messages object
    messages = compliance_prompt.format_messages(risk_summary=risk_summary)  # Fills in the {risk_summary} placeholder RIGHT NOW, giving the complete, real prompt as a plain list of messages (not a chain)

    # Loop: let the model call tools as many times as it needs
    while True:
        response = model_with_tools.invoke(messages)  # response is either a tool search request or the model's final answer
        messages.append(response)      #  # Add the model's reply to the conversation history

        if not response.tool_calls:
            break  # model is done searching, ready to give final answer


        # response.tool_calls is a list of dictionaries. Each item in that list represents one tool request the model made. 
        # If the model asked for 2 searches, this list would have 2 items in it. If the model made no tool requests (final answer), this list is empty [].


        # response.tool_calls is a list (possibly containing 1 or more search requests). This loop goes through each request, one at a time — calling each one tool_call while we're working on it.
        # response.tool_calls is a list of dictionaries

        for tool_call in response.tool_calls:                # The model might ask for multiple searches at once; loop through each request               
            result = search_tool.invoke(tool_call["args"])  # args comes from response.tool_calls - the query the model requested
            messages.append({
                "role": "tool",                             # marks this message as coming from a tool, not human/AI
                "content": result,                          # the actual search result text we got from tool
                "tool_call_id": tool_call["id"]             # links this result back to the model's original request
            })

    # Parse the final response into ComplianceStatus
    parser = PydanticOutputParser(pydantic_object=ComplianceStatus)
    final_result = parser.parse(response.content)                         # validates the MODEL's final answer against ComplianceStatus schema

    return final_result, tool_call['args']

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------





def run_agent2(risk_assessment: RiskAssessment) -> ComplianceStatus:
    docs = load_regulatory_docs()
    chunks = split_documents(docs)
    vector_store = create_regulatory_vector_store(chunks)
    retriever = create_retriever(vector_store)

    compliance_status = run_compliance_analysis(retriever, risk_assessment)
    compliance_status.case_id = risk_assessment.case_id  # overwrite with reliable case_id from Agent 1

    return compliance_status




def run_agent2_tool_calling(risk_assessment: RiskAssessment) -> ComplianceStatus:
    docs = load_regulatory_docs()
    chunks = split_documents(docs)
    vector_store = create_regulatory_vector_store(chunks)
    retriever = create_retriever(vector_store)

    compliance_status, search_query = run_compliance_analysis_tool_calling(retriever, risk_assessment)
    compliance_status.case_id = risk_assessment.case_id

    return compliance_status, search_query



if __name__ == "__main__":
    from agents.agent1_risk import run_agent1

    FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-309.pdf')
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
    print("AGENT 2 OUTPUT - SIMPLE VERSION (fixed 3 queries)")
    print("="*80)
    print(compliance_status)


    print("\n\n" + "="*80)
    print("AGENT 2 OUTPUT - TOOL-CALLING VERSION")
    print("="*80)

    compliance_status_tool_calling, search_query = run_agent2_tool_calling(risk_assessment)
    print(compliance_status_tool_calling)
    print()
    print(f"  [Tool Call] Model searched for: {search_query}")
