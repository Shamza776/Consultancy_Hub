from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize the same embeddings used in ingestion
embeddings = OllamaEmbeddings(model="nomic-embed-text")

@tool
def legal_specialist_tool(query: str) -> str:
    """Consult this tool for any questions related to company legal policies, 
    NDAs, intellectual property, data privacy, or contract protocols. 
    Input should be a specific search query."""
    
    # Connect to the local database you just created
    vector_db = Chroma(
        persist_directory="Database/legal_db",
        embedding_function=embeddings,
        collection_name="legal_specialist"
    )
    
    # Search for the top 2 most relevant segments
    results = vector_db.similarity_search(query, k=2)
    
    # Combine the results into a single string for the LLM to read
    return "\n\n".join([doc.page_content for doc in results])
@tool
def hr_specialist_tool(query: str) -> str:
    """Consult this for HR, benefits, and onboarding. (MOCK)"""
    return "HR Agent: I'm currently under development, but I will eventually handle HR queries!"

@tool
def it_specialist_tool(query: str) -> str:
    """Consult this for IT support and troubleshooting. (MOCK)"""
    return "IT Agent: I'm currently under development, but I will eventually handle IT queries!"

@tool
def customer_success_tool(query: str) -> str:
    """Consult this for customer retention and support logs. (MOCK)"""
    return "CS Agent: I'm currently under development, but I will eventually handle CS queries!"

