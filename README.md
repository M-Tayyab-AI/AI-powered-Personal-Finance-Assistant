# AI-powered-Personal-Finance-Assistant

An AI assistant focused on personal finance, budgeting guidance, and transaction-based analysis using a tool-calling LLM workflow.

## System Architecture Overview

This project uses a modular architecture with clear separation between orchestration, LLM configuration, prompts, API surface, and data schema:

- `api.py`
  - Exposes FastAPI endpoints:
    - `GET /health` for health checks
    - `GET /transactions` for user transaction retrieval
    - `POST /chat/` for finance assistant chat
  - Routes chat requests to the core agent function in `main.py`.

- `main.py`
  - Core orchestration layer for the assistant.
  - Maintains per-user conversation history.
  - Builds a LangChain tool-calling agent with two tools:
    - `financial_advice_tool` (RAG-based guidance)
    - `transactions_tool` (user transaction analysis)

- `llm.py`
  - Central LLM and embeddings configuration.
  - Provides:
    - Gemini chat model initialization
    - Voyage embeddings initialization
    - FAISS vector store loading with caching

- `prompts.py`
  - Contains system prompts for:
    - Main assistant behavior
    - RAG advice generation
    - Transaction analytics responses

- `schema.py`
  - Pydantic request/response models used across API and chat flow.

- Data folders
  - `faiss_index/` for knowledge base retrieval index
  - `rag_data/` for financial knowledge source content
  - `users_financial_data/` for per-user transaction files

## Design Decisions

- **Tool-calling agent design**  
  The assistant decides whether to answer directly, retrieve knowledge (`financial_advice_tool`), or analyze real transaction data (`transactions_tool`).

- **RAG for financial knowledge**  
  Advisory responses are grounded in FAISS-retrieved financial documents to reduce hallucinations and improve relevance.

- **Data-driven transaction analysis**  
  Transaction questions are answered using user transaction records fetched via API, then summarized by the LLM.

- **Per-user conversation memory**  
  In-memory chat history is keyed by `user_id` so each user context is kept separate.

- **Config-driven behavior**  
  Model names, temperatures, retrieval depth, and ports are centralized in `constants.py`.

## Setup Instructions

### 1) Clone the repository

```bash
git clone https://github.com/M-Tayyab-AI/AI-powered-Personal-Finance-Assistant.git
cd AI-powered-Personal-Finance-Assistant
```

### 2) Create and activate a virtual environment (recommended)

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in the project root and set required keys:

- `GEMINI_API_KEY`
- `VOYAGE_API_KEY`
- Optional runtime variables such as `PORT`

### 5) Run the API

```bash
python api.py
```

The FastAPI service runs on the configured port (default in code: `8000`).

## Example Queries

You can use prompts like:

- "What is the 50/30/20 rule and how do I apply it to my monthly income?"
- "Explain zero-based budgeting and give me an example monthly plan."
- "What is the Rule of 72 and how does compound interest work?"
- "How much should I be saving based on my age?"
- "What is the difference between an asset and a liability?"
- "How do I calculate my net worth?"

## Notes

- Keep `faiss_index/` available for RAG-based finance guidance.
- Ensure user transaction files exist in `users_financial_data/` for transaction questions.