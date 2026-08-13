"""
Configuration management for the Conversational RAG system.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves secret from Streamlit Cloud Secrets (st.secrets) first,
    then falls back to environment variables / .env.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


@dataclass
class RAGConfig:
    # Groq LLM Settings
    groq_api_key: Optional[str] = field(
        default_factory=lambda: get_secret("GROQ_API_KEY")
    )
    groq_model: str = field(
        default_factory=lambda: get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    temperature: float = field(
        default_factory=lambda: float(get_secret("LLM_TEMPERATURE", "0.2"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(get_secret("LLM_MAX_TOKENS", "2048"))
    )

    # Embedding Settings
    embedding_model_name: str = field(
        default_factory=lambda: get_secret(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Vector Store Settings
    vector_store_type: str = field(
        default_factory=lambda: get_secret("DEFAULT_VECTOR_STORE", "faiss").lower()
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: get_secret("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    faiss_persist_dir: str = field(
        default_factory=lambda: get_secret("FAISS_PERSIST_DIR", "./data/faiss_index")
    )

    # Pinecone Settings
    pinecone_api_key: Optional[str] = field(
        default_factory=lambda: get_secret("PINECONE_API_KEY")
    )
    pinecone_index_name: str = field(
        default_factory=lambda: get_secret("PINECONE_INDEX_NAME", "langchain-rag-index")
    )
    pinecone_environment: str = field(
        default_factory=lambda: get_secret("PINECONE_ENVIRONMENT", "us-east-1")
    )

    # Chunking & Retrieval Parameters
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "4"))
    )


# Available Model Options
AVAILABLE_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
]

AVAILABLE_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]

SUPPORTED_VECTOR_STORES = ["faiss", "chroma", "pinecone"]
