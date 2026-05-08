import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings

from constants import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    VOYAGE_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
)

load_dotenv()

#  LLM Factory

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns a configured Gemini 2.5 Flash LLM instance via LangChain.
    GEMINI_API_KEY is read from the .env file.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Please add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        google_api_key=api_key,
        convert_system_message_to_human=False,  # Gemini supports system messages natively
    )

#  Jina

def get_embeddings() -> VoyageAIEmbeddings:
    """
    Returns a JinaEmbeddings instance using the model defined in constants.
    VOYAGE_API_KEY is read from the .env file.
    """
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY is not set. "
            "Please add it to your .env file."
        )

    return VoyageAIEmbeddings(
        voyage_api_key=api_key,
        model=VOYAGE_EMBEDDING_MODEL,
    )


#  FAISS Vector Store Loader  (cached so it loads only once)

@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """
    Loads and returns the FAISS vector store from disk.

    NOTE:
    - Call this once; lru_cache ensures it is only loaded once per process.

    Raises:
        FileNotFoundError: If the FAISS index directory does not exist.
        EnvironmentError:  If JINA_API_KEY is missing.
    """
    index_path = FAISS_INDEX_PATH

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Please build the vector store first and place it at that path. "
            "Expected files: faiss_index/index.faiss and faiss_index/index.pkl"
        )

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,   # required for FAISS pickle files
    )

    return vectorstore