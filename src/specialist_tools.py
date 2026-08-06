from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the same embeddings used in ingestion.
# Runs locally on CPU (no external server / API key needed) — this is what
# makes it deployable on a free host, unlike OllamaEmbeddings which needs
# a running Ollama server.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _query_department(query: str, db_path: str, collection_name: str, k: int = 2) -> str:
    """Shared retrieval logic for all department tools."""
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    results = vector_db.similarity_search(query, k=k)
    if not results:
        return "No relevant information was found in the internal database for this query."
    return "\n\n".join([doc.page_content for doc in results])


@tool
def legal_specialist_tool(query: str) -> str:
    """Consult this tool for any questions related to company legal policies, 
    NDAs, intellectual property, data privacy, or contract protocols. 
    Input should be a specific search query."""
    return _query_department(query, "Database/legal_db", "legal_specialist")


@tool
def hr_specialist_tool(query: str) -> str:
    """Consult this tool for questions about HR policy, including leave,
    benefits, onboarding, grievances, and remote work.
    Input should be a specific search query."""
    return _query_department(query, "Database/hr_db", "hr_specialist")


@tool
def it_specialist_tool(query: str) -> str:
    """Consult this tool for questions about IT support, security policy,
    acceptable use, hardware requests, and incident reporting.
    Input should be a specific search query."""
    return _query_department(query, "Database/it_db", "it_specialist")


@tool
def customer_success_tool(query: str) -> str:
    """Consult this tool for questions about customer support SLAs,
    escalation procedures, refund policy, and churn prevention.
    Input should be a specific search query."""
    return _query_department(query, "Database/customer_success_db", "customer_success_specialist")