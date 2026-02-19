import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURATION ---
# Path to your generated legal PDF
DATA_PATH = "Data/legal_compliance_handbook.pdf"
# Path where the database will be saved (ignored by git)
DB_PATH = "Database/legal_db"
# The name of the collection inside Chroma
COLLECTION_NAME = "legal_specialist"
# The local embedding model we agreed on
EMBEDDING_MODEL = "nomic-embed-text"

def run_legal_ingestion():
    print(f"--- Legal Specialist: Starting Ingestion ---")

    # 1. Check if the PDF exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please add the PDF to the Data folder.")
        return

    # 2. LOAD: Extract text from the PDF
    loader = PyPDFLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Successfully loaded {len(raw_documents)} pages from PDF.")

    # 3. SPLIT: Break text into chunks for better retrieval
    # chunk_overlap ensures the AI doesn't lose context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(raw_documents)
    print(f"Documents split into {len(docs)} chunks.")

    # 4. EMBED & STORE: Create the local Vector Database
    print(f"Generating embeddings using '{EMBEDDING_MODEL}'... (This may take a minute)")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # This creates the vectorstore and saves it to the Database/ folder
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print(f"--- SUCCESS: Legal Database created at {DB_PATH} ---")

if __name__ == "__main__":
    run_legal_ingestion()
