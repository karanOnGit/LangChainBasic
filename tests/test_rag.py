"""
Automated Test Suite for Conversational RAG system.
Tests loaders, splitters, embeddings, vector stores, memory, and chain logic.
"""

import os
import shutil
import pytest
from pathlib import Path

from src.loaders import DocumentLoaderManager
from src.splitters import TextSplitterManager
from src.embeddings import get_embedding_model
from src.vectorstores import VectorStoreManager
from src.memory import MemoryManager
from langchain_core.documents import Document


@pytest.fixture
def sample_text():
    return (
        "Retrieval-Augmented Generation (RAG) is an AI framework for retrieving relevant facts "
        "from an external knowledge base to evaluate and answer queries accurately. "
        "Groq provides ultra-high-speed inference with LPUs. "
        "Vector stores like FAISS, Chroma, and Pinecone store embeddings for fast similarity search."
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    d = tmp_path / "test_data"
    d.mkdir()
    return d


def test_document_loader_from_text(sample_text):
    docs = DocumentLoaderManager.load_from_text(sample_text, source_name="test_doc.txt")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "test_doc.txt"
    assert "Retrieval-Augmented Generation" in docs[0].page_content


def test_document_loader_from_file(temp_data_dir, sample_text):
    file_path = temp_data_dir / "sample.txt"
    file_path.write_text(sample_text, encoding="utf-8")

    docs = DocumentLoaderManager.load_from_file(str(file_path))
    assert len(docs) >= 1
    assert docs[0].metadata["file_name"] == "sample.txt"
    assert docs[0].metadata["file_type"] == "txt"


def test_document_loader_from_upload(sample_text):
    raw_bytes = sample_text.encode("utf-8")
    docs = DocumentLoaderManager.load_from_upload("upload_sample.txt", raw_bytes)
    assert len(docs) >= 1
    assert docs[0].metadata["source"] == "upload_sample.txt"


def test_text_splitter():
    long_text = "Sentence one about AI. " * 50
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})

    splitter = TextSplitterManager(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_index"] == i
        assert len(chunk.page_content) <= 150


def test_embeddings_generation():
    embeddings = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vec = embeddings.embed_query("Hello world RAG")
    assert isinstance(vec, list)
    assert len(vec) == 384  # MiniLM dimension is 384


def test_faiss_vectorstore_create_and_search(sample_text, temp_data_dir):
    docs = DocumentLoaderManager.load_from_text(sample_text, source_name="faiss_test")
    splitter = TextSplitterManager(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    faiss_dir = str(temp_data_dir / "faiss_test_index")

    vs = VectorStoreManager.create_vector_store(
        store_type="faiss",
        documents=chunks,
        embeddings=embeddings,
        persist_dir=faiss_dir,
    )

    assert vs is not None
    # Test retrieval
    retriever = VectorStoreManager.get_retriever(vs, k=2)
    results = retriever.invoke("What does Groq provide?")
    assert len(results) > 0
    assert any("Groq" in r.page_content for r in results)

    # Test load existing
    loaded_vs = VectorStoreManager.load_vector_store(
        store_type="faiss",
        embeddings=embeddings,
        persist_dir=faiss_dir,
    )
    assert loaded_vs is not None


def test_chroma_vectorstore_create_and_search(sample_text, temp_data_dir):
    docs = DocumentLoaderManager.load_from_text(sample_text, source_name="chroma_test")
    splitter = TextSplitterManager(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    chroma_dir = str(temp_data_dir / "chroma_test_index")

    vs = VectorStoreManager.create_vector_store(
        store_type="chroma",
        documents=chunks,
        embeddings=embeddings,
        persist_dir=chroma_dir,
        collection_name="test_collection",
    )

    assert vs is not None
    retriever = VectorStoreManager.get_retriever(vs, k=2)
    results = retriever.invoke("Which vector stores are mentioned?")
    assert len(results) > 0


def test_memory_manager():
    memory = MemoryManager(max_history_messages=4)
    session_id = "test_session_abc"

    hist = memory.get_session_history(session_id)
    hist.add_user_message("Hello")
    hist.add_ai_message("Hi there!")
    hist.add_user_message("What is RAG?")
    hist.add_ai_message("RAG is Retrieval-Augmented Generation.")
    hist.add_user_message("Tell me more.")
    hist.add_ai_message("It combines search with LLMs.")

    # Check pruning to max 4 messages
    hist_pruned = memory.get_session_history(session_id)
    assert len(hist_pruned.messages) <= 4

    formatted = memory.get_formatted_history(session_id)
    assert len(formatted) <= 4
    assert formatted[-1]["role"] == "assistant"

    # Test clear session
    memory.clear_session(session_id)
    assert len(memory.get_session_history(session_id).messages) == 0
