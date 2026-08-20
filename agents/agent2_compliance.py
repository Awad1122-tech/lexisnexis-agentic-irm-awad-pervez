### Agent 2: Compliance Agent

"""
RAG PIPELINE OVER REGULATORY DOCUMENTS (GDPR, FCA, AML/KYC: CDD/EDD/PEP).
Takes Agents 1's RiskAssessment and produces structured ComplianceStatus.
Single entry point: run_agent2(risk_assessment) -> ComplianceStatus

"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
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
         assessment findings, determine this case's compliance status:

        Retrieved Regulatory Context:
        {context}

        Case Risk Assessment Summary:
        {risk_summary}

        1. GDPR compliance: is the required safeguard (Article 22C - information,
           representations, human intervention, right to contest) in place given whether
           human review is required for this case?
        2. FCA compliance: does this case demonstrate the accountability, fairness, and
           explainability the FCA requires?
        3. AML/KYC compliance: if a PEP match was identified, was enhanced due diligence
           actually completed? If not, this is non-compliant per Regulation 35.
        4. Mortality-confirmed block: if the case shows a CONFIRMED mortality match, all
           compliance fields must be False and the case must be flagged as non-compliant,
           regardless of any other findings.
        5. List specific compliance flags found (in plain language, grounded in the
           retrieved regulatory text).
        6. List specific regulatory actions required to bring the case into compliance.

        {format_instructions}""")
    ])

    return compliance_prompt.partial(format_instructions=format_instructions)



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


if __name__ == "__main__":
    from agents.agent1_risk import run_agent1

    FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-301.pdf')
    risk_assessment = run_agent1(FILE_PATH)
    compliance_status = run_agent2(risk_assessment)
    print(compliance_status.case_id)
    print(compliance_status)