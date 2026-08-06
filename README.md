# 🏢 Consultancy Hub

An AI-powered corporate assistant that answers employee questions by routing them
to specialized department agents — Legal, HR, IT, and Customer Success — each
backed by its own Retrieval-Augmented Generation (RAG) pipeline over real
department documents.

Built as a hands-on exploration of agentic systems: an LLM-driven manager agent
decides which specialist tool to call based on the nature of the question, then
retrieves grounded answers from a per-department vector database instead of
hallucinating policy details.

---

## ✨ Features

- **Multi-agent orchestration** — a manager agent (built with LangGraph's
  `create_react_agent`) reasons about each query and routes it to the correct
  department tool.
- **Department-specific RAG** — Legal, HR, IT, and Customer Success each have
  their own Chroma vector store, built from real policy documents, so answers
  are grounded in actual source text rather than the model's general knowledge.
- **Local embeddings** — uses `sentence-transformers` (`all-MiniLM-L6-v2`)
  running entirely in-process, so no external embedding API or key is required.
- **Cloud LLM via OpenRouter** — the reasoning model runs on OpenRouter's free
  tier, making the whole system deployable without any paid infrastructure.
- **Streamlit chat interface** — a simple, familiar chat UI with conversation
  history and a live department status sidebar.

---

## 🧠 Architecture

```
User question
      │
      ▼
Streamlit UI (ui.py)
      │
      ▼
Manager Agent (main_orchestrator.py)
 — LangGraph ReAct agent, decides which tool to call
      │
      ├── legal_specialist_tool ──► Chroma (Database/legal_db)
      ├── hr_specialist_tool ─────► Chroma (Database/hr_db)
      ├── it_specialist_tool ─────► Chroma (Database/it_db)
      └── customer_success_tool ──► Chroma (Database/customer_success_db)
              │
              ▼
      Retrieved context ──► LLM (OpenRouter) ──► Final answer
```

---

## 🛠️ Tech Stack

| Layer            | Technology                                    |
|-------------------|------------------------------------------------|
| UI                | Streamlit                                       |
| Orchestration     | LangGraph (`create_react_agent`)                |
| LLM               | OpenRouter (free-tier models, OpenAI-compatible)|
| Embeddings        | `sentence-transformers` (local, CPU)            |
| Vector Store      | Chroma                                          |
| Document Loading  | LangChain (`PyPDFLoader`, `TextLoader`)         |

---

## 📁 Project Structure

```
Consultancy_Hub/
├── Data/                       # Source department documents (PDF/TXT)
├── Database/                   # Per-department Chroma vector stores (generated)
├── src/
│   ├── ingest_all.py           # Builds all department vector stores
│   └── specialist_tools.py     # Department RAG tools used by the agent
├── main_orchestrator.py        # Manager agent — LLM + tool routing
├── ui.py                       # Streamlit chat interface
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and set up the environment
```bash
git clone https://github.com/<your-username>/Consultancy_Hub.git
cd Consultancy_Hub
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key
Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys), then:
```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```
(Optional) choose a specific free model:
```bash
export OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```
Check [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0)
for currently available free models.

### 3. Build the department knowledge bases
Run this once (and again any time source documents change):
```bash
python src/ingest_all.py
```

### 4. Run the app
```bash
streamlit run ui.py
```

---

## 💬 Example Questions

- *"What's the policy on facilitation payments?"* (Legal)
- *"How many sick days do I get?"* (HR)
- *"What's the response time for a critical IT ticket?"* (IT)
- *"When do refunds need Director approval?"* (Customer Success)

---

## 🗺️ Roadmap

- [ ] Replace mock HR/IT/Customer Success source docs with real internal
      documentation
- [ ] Add conversation memory across sessions
- [ ] Add authentication for multi-user deployments
- [ ] Deploy to Streamlit Community Cloud

---

## 📄 License

This project is for educational/portfolio purposes.
