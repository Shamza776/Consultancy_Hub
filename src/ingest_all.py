import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Local embedding model — runs on CPU, no server/API key needed.
# Must match the model used in specialist_tools.py.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# One entry per department. Add more here as new departments come online.
DEPARTMENTS = [
    {
        "name": "Legal",
        "data_path": "Data/legal_compliance_handbook.pdf",
        "loader": "pdf",
        "db_path": "Database/legal_db",
        "collection_name": "legal_specialist",
    },
    {
        "name": "HR",
        "data_path": "Data/hr_policies.txt",
        "loader": "text",
        "db_path": "Database/hr_db",
        "collection_name": "hr_specialist",
    },
    {
        "name": "IT",
        "data_path": "Data/it_support_guide.txt",
        "loader": "text",
        "db_path": "Database/it_db",
        "collection_name": "it_specialist",
    },
    {
        "name": "Customer Success",
        "data_path": "Data/customer_success_playbook.txt",
        "loader": "text",
        "db_path": "Database/customer_success_db",
        "collection_name": "customer_success_specialist",
    },
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def run_ingestion(dept: dict):
    print(f"\n--- {dept['name']} Specialist: Starting Ingestion ---")

    if not os.path.exists(dept["data_path"]):
        print(f"Error: {dept['data_path']} not found. Skipping {dept['name']}.")
        return

    if dept["loader"] == "pdf":
        loader = PyPDFLoader(dept["data_path"])
    else:
        loader = TextLoader(dept["data_path"], encoding="utf-8")

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} document(s) from {dept['data_path']}.")

    docs = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(docs)} chunks.")

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=dept["db_path"],
        collection_name=dept["collection_name"],
    )

    print(f"--- SUCCESS: {dept['name']} database created at {dept['db_path']} ---")


if __name__ == "__main__":
    for department in DEPARTMENTS:
        run_ingestion(department)

    print("\n--- All department ingestion complete ---")