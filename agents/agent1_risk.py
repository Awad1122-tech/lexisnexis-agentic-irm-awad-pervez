from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from schemas import RiskAssessment
import os
import hashlib

load_dotenv()

# Data file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'CASE-2026-301.pdf')


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


# Reliably extract the real case ID directly from the text using, instead of trusting the LLM to correctly report it. Used for deduplication and to override the LLM's own guess later.

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

def get_content_hash(docs) -> str:
    full_text = "\n".join(doc.page_content for doc in docs)
    return hashlib.md5(full_text.encode()).hexdigest()




## Step 3 of RAG Pipeline: vector store creation (Embeddings and ChromaDB)

def create_vector_store(chunks, case_id: str, content_hash: str):

    # convert chunks to embeddings and store in ChromaDB
    # Each case is identified by its unuique case_id to prevent duplication

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

    """
    Keeps ChromaDB in sync with the actual case file, handling 3 scenarios:
    1. Brand new case (never embedded before) -> embed it
    2. Same case, unchanged content -> skip entirely, avoids duplicate embeddings
    3. Same case, content changed (e.g. PDF was edited/fixed) -> delete the old
       stale version first, then embed the corrected version fresh

    Uses a content hash (fingerprint of the text) instead of just case_id alone,
    so edits to a case file are automatically detected and refreshed - no manual
    deletion needed, and no risk of stale or duplicate data lingering in the database.
    """

    needs_refresh = True
    if len(existing['ids']) > 0:
        existing_hash = existing['metadatas'][0].get('content_hash')
        if existing_hash == content_hash:
            needs_refresh = False

    if needs_refresh:
        vector_store.delete(where={"case_id": case_id})
        for chunk in chunks:
            chunk.metadata['case_id'] = case_id
            chunk.metadata['content_hash'] = content_hash
        vector_store.add_documents(chunks)                 # vector_store.add_documents(chunks) = embed (using embedding_model) + store (into ChromaDB), both happening together, in that exact order, within this single line.

    return vector_store




 # vector_store.add_documents(chunks) = embed (using embedding_model) + store (into ChromaDB), both happening together, in that exact order, within this single line.


## Step 4 of RAG Pipeline: Creating retriever to search ChromaDB based similarity

def create_retriever(vector_store, case_id: str):

    retriever = vector_store .as_retriever(      # .as_retriever() is a method that converts your vector_store (the Chroma database object) into a proper Retriever object - which is a Runnable, meaning it can be used with .invoke() and chained with |.
        search_type = "similarity",
        search_kwargs = {
            "k" : 12,
            "filter": {"case_id": case_id}  # This restricts the similarity search to ONLY chunks that belong to the current case, it's not part of the similarity calculation itself, it's a hard restriction applied first, narrowing the search pool before comparing embeddings.

        } 
    )

    return retriever


## Step 5 of RAG Pipeline: Augmentation (creating prompt template, combining context and question), builds the prompt template that will later be combined with retrieved context and sent to the LLM.


# This function only does JOB 1 (telling the LLM the expected format through format_instructions), it does NOT validate anything yet. Validation (JOB 2) happens later, in run_risk_analysis(), when the LLM's actual response comes back.

def create_prompt():

    # format_instructions, generated by get_format_instructions() and inserted in the prompt text via .partial() at the end:
    # It's purpose is like this: "please respond using this JSON structure." The LLM reads this instruction, and (hopefully) 
    # follows it when generating its response - but there's no guarantee it will, since it's just a text instruction, not an enforced rule at this stage.


    parser = PydanticOutputParser(pydantic_object=RiskAssessment)  # Create a parser tied to the RiskAssessment schema
    format_instructions = parser.get_format_instructions()

    # ChatPromptTemplate is LangChain's tool for building a reusable prompt with placeholders (like {context}, {question}) that get filled in with real values later, when the chain actually runs.

    # .from_messages() is a method of the ChatPromptTemplate class used to construct/create a ChatPromptTemplate object from a list of role-based messages.

    risk_prompt = ChatPromptTemplate.from_messages([
        ("system", """ You are an expert identity risk analyst at LexisNexis Risk Solution. 
         You analyse identity verification, fraud screening, and identity tracing data from
         LexisNexis products: ThreatMetrix, IDU / Instant Verify International, TrueID, TraceIQ, and Bridger Insight XG.
         You provide detailed, accurate, and professionally structured risk assessments.
         Always base your analysis strictly on the provided context.
         Never make assumptions beyond what the data shows."""),


         ("human", """ Based on the following LexisNexis case data, provide a comprehensive risk analysis: 
         
         
        Context:
        {context}


        Analysis required: {question}



         1. Overall risk rating (LOW / MEDIUM / MEDIUM-HIGH / HIGH)
         2. Key signals identified (short list, across ALL products including TraceIQ)
         3. ThreatMetrix: trust score, risk score
         4. Identity verification: overall result from IDU/Instant Verify, document type submitted,
            document authentication result and score, facial match score, liveness check result
         5. Whether a PEP match was identified - if true, extract the source list, matched
            record, and match basis from the Bridger Insight XG section
         6. Whether a sanctions match was identified - if true, extract the same evidence detail
         7. Adverse media findings - for each article found, extract its description and the
            evidence type/source (e.g. "National news archive search"); and State-Owned Enterprise flag
         8. TraceIQ findings: LexID, aliases, current address with residency score and supporting
            source records, property ownership details, occupants and how they were sourced,
            previous address history with source records, mortality status, ALL CCJs found
            (with court, date, and amount for each), bankruptcy/IVA/DRO status, and company
            directorships
        9.  A detailed written explanation of each risk signal - several sentences per signal,
             not just a label. Where the case data provides specific clarifying detail (e.g., the
             exact source of a PEP or State-Owned Enterprise connection, whether it applies to the
             subject directly or via an associated individual such as a spouse), explicitly state
             that detail rather than summarizing generically. When describing numerical scores
             (e.g., facial match, document authentication, trust, risk), ensure your characterisation
             (high/low, strong/weak) is consistent with the actual score value provided in the
             context. Put this in the detailed_analysis field
         10. Recommended action and suggested next steps
         11. Regulatory considerations (GDPR, FCA, AML/KYC) - your own general assessment
         12. Confidence level and confidence score (0-100) representing your certainty in
             this assessment
         
        {format_instructions}""")

    ])

    return risk_prompt.partial(format_instructions=format_instructions) # .partial(format_instructions=format_instructions)'s exact purpose is to lock in the {format_instructions} placeholder with its real value, right now, permanently, so it never needs to be provided again later, at .invoke() time.



## Step 6 of RAG Pipeline: Generation (Generating Response using LLM) using chains

def run_risk_analysis(retriever, question: str):

    # Model:

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    max_new_tokens=2000
    )

    model = ChatHuggingFace(llm=llm)

    # Job 2 - CHECKING the LLM's actual response (happens AFTER the LLM responds):

    """
    This is where parser (the PydanticOutputParser object) actually does its real work - after the LLM has already generated text back. 
    The parser takes that raw text, tries to interpret it as JSON, and then runs it through Pydantic's validation (checking types, required fields, etc.)
    
    """

    parser = PydanticOutputParser(pydantic_object=RiskAssessment)
    risk_prompt = create_prompt()

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)


    """
    risk_prompt is built using ChatPromptTemplate.from_messages([]) - and ChatPromptTemplate is one of LangChain's built-in Runnable classes. It automatically comes with .invoke() and supports the | chaining operator
    
    """

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
    content_hash = get_content_hash(docs)
    vector_store = create_vector_store(chunks, case_id, content_hash)
    retriever = create_retriever(vector_store, case_id)

    question = "Provide a comprehensive risk analysis of this identity verification case, covering fraud signals, identity verification, document authentication, compliance screening, and identity tracing findings."

    risk_assessment = run_risk_analysis(retriever, question)
    risk_assessment.case_id = case_id                          # overwrite LLM's guess with reliably extracted case_id
    risk_assessment.adverse_media_count = len(risk_assessment.adverse_media_articles)  # 
    # overwrite LLM's count with the actual list length, guaranteed accurate

    return risk_assessment



if __name__ == "__main__":
    result = run_agent1(FILE_PATH)   # here, the global FILE_PATH is passed IN as the argument
    print(result.case_id)
    print(result)
