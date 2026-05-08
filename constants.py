#  LLM
LLM_MODEL_NAME        = "gemini-2.5-flash"
LLM_TEMPERATURE       = 0.2
# LLM_MAX_OUTPUT_TOKENS = 2048

#  Voyage
VOYAGE_EMBEDDING_MODEL = "voyage-4"

#  FAISS Vector Store
FAISS_INDEX_PATH = "faiss_index"   # folder that holds index.faiss + index.pkl
RAG_RETRIEVER_K  = 20               # top-k docs to retrieve

#  Mock Banking API
BANKING_API_HOST        = "http://localhost"
BANKING_API_PORT        = 8000
BANKING_API_BASE_URL    = f"{BANKING_API_HOST}:{BANKING_API_PORT}"
TRANSACTIONS_ENDPOINT   = "/transactions"

#  Chat Agent API
CHAT_APP_HOST = "0.0.0.0"
CHAT_APP_PORT = 8001

# ─────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────
AGENT_VERBOSE         = True
MAX_AGENT_ITERATIONS  = 5