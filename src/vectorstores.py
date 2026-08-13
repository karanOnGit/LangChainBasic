"""
Unified Vector Store Manager.
Supports FAISS, ChromaDB, and Pinecone with seamless switching and persistence.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

# Vector store implementations
from langchain_community.vectorstores import FAISS

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        Chroma = None

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Factory and manager for local and cloud vector databases."""

    SUPPORTED_STORES = ["faiss", "chroma", "pinecone"]

    @classmethod
    def create_vector_store(
        cls,
        store_type: str,
        documents: List[Document],
        embeddings: Embeddings,
        persist_dir: Optional[str] = None,
        collection_name: str = "langchain_rag",
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
    ) -> VectorStore:
        """
        Creates a new vector store from document chunks.
        """
        store_type = store_type.lower().strip()
        if not documents:
            raise ValueError("Cannot initialize VectorStore with empty document list.")

        logger.info(
            f"Building {store_type.upper()} vector store with {len(documents)} document chunks..."
        )

        if store_type == "faiss":
            vector_store = FAISS.from_documents(documents, embeddings)
            if persist_dir:
                os.makedirs(persist_dir, exist_ok=True)
                vector_store.save_local(persist_dir)
                logger.info(f"FAISS index persisted to '{persist_dir}'")
            return vector_store

        elif store_type == "chroma":
            if Chroma is None:
                raise ImportError(
                    "langchain-chroma or chromadb must be installed to use Chroma vector store. "
                    "Please install `langchain-chroma`."
                )
            if persist_dir:
                os.makedirs(persist_dir, exist_ok=True)
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=persist_dir,
                    collection_name=collection_name,
                )
                logger.info(f"ChromaDB persisted to '{persist_dir}'")
            else:
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=collection_name,
                )
            return vector_store

        elif store_type == "pinecone":
            try:
                from langchain_pinecone import PineconeVectorStore
                from pinecone import Pinecone
            except ImportError:
                raise ImportError(
                    "pinecone and langchain-pinecone must be installed to use Pinecone vector store."
                )

            api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
            index_name = pinecone_index_name or os.getenv("PINECONE_INDEX_NAME")

            if not api_key:
                raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY in .env.")
            if not index_name:
                raise ValueError("Pinecone index name is required. Set PINECONE_INDEX_NAME in .env.")

            # Initialize Pinecone client
            pc = Pinecone(api_key=api_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if index_name not in existing_indexes:
                logger.warning(
                    f"Pinecone index '{index_name}' not found. Please ensure it exists in your Pinecone dashboard."
                )

            vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=index_name,
                pinecone_api_key=api_key,
            )
            return vector_store

        else:
            raise ValueError(
                f"Unsupported vector store '{store_type}'. Choose from {cls.SUPPORTED_STORES}"
            )

    @classmethod
    def load_vector_store(
        cls,
        store_type: str,
        embeddings: Embeddings,
        persist_dir: Optional[str] = None,
        collection_name: str = "langchain_rag",
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
    ) -> Optional[VectorStore]:
        """
        Loads an existing persisted vector store if available.
        """
        store_type = store_type.lower().strip()

        if store_type == "faiss":
            if persist_dir and os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "index.faiss")):
                logger.info(f"Loading existing FAISS index from '{persist_dir}'")
                return FAISS.load_local(
                    persist_dir,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            return None

        elif store_type == "chroma":
            if Chroma is None:
                return None
            if persist_dir and os.path.exists(persist_dir):
                logger.info(f"Loading existing ChromaDB from '{persist_dir}'")
                return Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name,
                )
            return None

        elif store_type == "pinecone":
            try:
                from langchain_pinecone import PineconeVectorStore
            except ImportError:
                return None

            api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
            index_name = pinecone_index_name or os.getenv("PINECONE_INDEX_NAME")

            if api_key and index_name:
                return PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    pinecone_api_key=api_key,
                )
            return None

        return None

    @classmethod
    def get_retriever(
        cls,
        vector_store: VectorStore,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> VectorStoreRetriever:
        """
        Returns a configured retriever from a vector store.
        """
        search_kwargs = {"k": k}
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
