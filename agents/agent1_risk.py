from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from schemas import RiskAssessment
import os

load_dotenv()

# Data file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-302.pdf')


## Step 1 of RAG Pipeline: Load the data/ document
def load_case_data(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

## Step 2 of RAG Pipeline: Text Chunking (split loaded document into smaller chunks for embedding)

def split_documents(docs):

    # TODO: chunk_size=500 may be too small given TraceIQ's longer sections
    # and evidence-level detail (source records, screening hits). If retrieval
    # quality seems weak, test increasing chunk_size to 800-1000.

    """
    I am using RecursiveCharacterTextSplitter because, it tries multiple separators in order (paragraph breaks, then line breaks, then spaces) 
    before falling back to a hard cut - this respects the data's natural line-by-line field structure (e.g. "Trust Score: 91/100" on its own line) 
    instead of cutting a single field in half.
    
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(docs)

    return chunks


def get_case_id(docs) -> str:
    for doc in docs:
        for line in doc.page_content.split('\n'):
            if 'Case ID:' in line:
                case_id = line.split('Case ID:')[1].strip()
                return case_id
    return "UNKNOWN-CASE"



# Purpose of get_case_id():

"""
It reliably extract the real case ID directly from the PDF's text, nstead of relying on the LLM to correctly report it later.

Why we are using this here:

1) Used for deduplication - create_vector_store() needs a real case_id to check "has this case already been embedded?" before adding it again.

2) Used to override the LLM's own guess - the LLM might slightly misspell or reformat the case ID when it generates its structured output. 
   By extracting it directly from the PDF with plain code, you get a guaranteed accurate value

"""


## Step 3 of RAG Pipeline: vector store creation (Embeddings and ChromaDB)

def create_vector_store(chunks, case_id: str):

    # convert chunks to embeddings and store in ChromaDB
    #Each case is identified by its unuique case_id to prevent duplication

    embedding_model = HuggingFaceEndpointEmbeddings(
        model = 'sentence-transformers/all-MiniLM-L6-v2',
        task = 'feature-extraction'
    )


    vector_store = Chroma(

        embedding_function = embedding_model,
        persist_directory = os.path.join(BASE_DIR, 'chroma_db'),
        collection_name = 'lexisnexis_cases'
    )


    # Check if this case_id has already been embedded, to avoid duplication

    existing = vector_store.get(where={"case_id": case_id})
    if len(existing['ids']) == 0:
        # Tag each chunk with case_id so future duplicate checks work
        for chunk in chunks:
            chunk.metadata['case_id'] = case_id
        vector_store.add_documents(chunks)

    return vector_store



## Step 4 of RAG Pipeline: Creating retriever to search ChromaDB based similarity

def create_retriever(vector_store, case_id: str):

    retriever = vector_store .as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k" : 12,
            "filter": {"case_id": case_id}

        } 
    )

    return retriever


## Step 5 of RAG Pipeline: Augmentation (creating prompt template, combining context and question)

def create_prompt():
    parser = PydanticOutputParser(pydantic_object=RiskAssessment)
    format_instructions = parser.get_format_instructions()

    risk_prompt = ChatPromptTemplate.from_messages([
        ("system", """ Your are an expert identity risk analyst at LexisNexis Risk Solution. 
         You analyse identity verification, fraud screening, and identity tracing data from
         LexisNexis products: ThreatMetrix, IDU / Instant Verify International, TrueID, TraceIQ, and Bridger Insight XG.
         You provide detailed, accurate, and professionally structured risk assessments.
         Always base you analysis your analysis strictly on the provided context.
         Never make assumptions beyond what the data shows."""),


         ("human", """ Based on the following LexisNexis case data, provide a comprehensive risk analysis: 
         
         
        Context:
        {context}


        Analysis required: {question}



         1. Overall risk rating (LOW / MEDIUM / MEDIUM-HIGH / HIGH)
         2. Key signals identified (short list, across ALL products including TraceIQ)
         3. ThreatMetrix: trust score, risk score
         4. Identity verification: overall result from IDU/Instant Verify, document authentication
            result and score, facial match score, liveness check result
         5. Whether a PEP match was identified - if true, extract the source list, matched
            record, and match basis from the Bridger Insight XG section
         6. Whether a sanctions match was identified - if true, extract the same evidence detail
         7. Adverse media count and State-Owned Enterprise flag
         8. TraceIQ findings: LexID, aliases, current address with residency score and supporting
            source records, property ownership details, occupants and how they were sourced,
            previous address history with source records, mortality status, ALL CCJs found
            (with court, date, and amount for each), bankruptcy/IVA/DRO status, and company
            directorships
         9. A detailed written explanation of each risk signal - several sentences per signal,
             not just a label. Put this in the detailed_analysis field.
         10. Recommended action and suggested next steps
         11. Regulatory considerations (GDPR, FCA, AML/KYC) - your own general assessment
         12. Confidence level and confidence score (0-100) representing your certainty in
             this assessment
         13. Whether enhanced due diligence has actually been COMPLETED for this case - look
             for evidence in the case data that EDD steps were carried out. If there is no
             such evidence, answer false.
         
        {format_instructions}""")

    ])

    return risk_prompt.partial(format_instructions=format_instructions)



## Step 6 of RAG Pipeline: Generation (Generating Response using LLM) using chains

def run_risk_analysis(retriever, question: str):

    # Model:

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    max_new_tokens=3000
    )

    model = ChatHuggingFace(llm=llm)


    parser = PydanticOutputParser(pydantic_object=RiskAssessment)
    risk_prompt = create_prompt()

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    chain = risk_prompt | model | parser

    result = chain.invoke({

        "context": context,
        "question": question
    })

    return result


def run_agent1(file_path: str) -> RiskAssessment:
    docs = load_case_data(file_path)
    chunks = split_documents(docs)
    case_id = get_case_id(docs)
    vector_store = create_vector_store(chunks, case_id)
    retriever = create_retriever(vector_store, case_id)

    question = "Provide a comprehensive risk analysis of this identity verification case, covering fraud signals, identity verification, document authentication, compliance screening, and identity tracing findings."

    risk_assessment = run_risk_analysis(retriever, question)
    risk_assessment.case_id = case_id                          # overwrite LLM's guess with reliably extracted case_id

    return risk_assessment



if __name__ == "__main__":
    result = run_agent1(FILE_PATH)   # here, the global FILE_PATH is passed IN as the argument
    print(result.case_id)
    print(result)
