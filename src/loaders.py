"""
Document Loading Module.
Provides robust multi-format document loading for PDF, DOCX, TXT, Markdown, and Web URLs.
"""

import os
import tempfile
from typing import List, Union, BinaryIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,
)


class DocumentLoaderManager:
    """Unified manager for loading documents from files, byte buffers, or URLs."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json"}

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> List[Document]:
        """Loads documents from a file path based on its extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        documents: List[Document] = []

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
                documents = loader.load()
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(path))
                documents = loader.load()
            elif ext in [".txt", ".md", ".csv", ".json"]:
                loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
                documents = loader.load()
            else:
                # Fallback to TextLoader
                loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
                documents = loader.load()

            # Enrich metadata
            for doc in documents:
                doc.metadata.setdefault("source", path.name)
                doc.metadata["file_name"] = path.name
                doc.metadata["file_type"] = ext.replace(".", "")
                doc.metadata["file_path"] = str(path.resolve())

            return documents

        except Exception as e:
            raise RuntimeError(f"Error loading document '{path.name}': {str(e)}") from e

    @classmethod
    def load_from_upload(
        cls, file_name: str, file_bytes: Union[bytes, BinaryIO]
    ) -> List[Document]:
        """
        Loads documents from an in-memory byte stream (e.g., Streamlit FileUploader)
        by creating a temporary file and processing it with standard loaders.
        """
        ext = Path(file_name).suffix.lower()
        if isinstance(file_bytes, (bytes, bytearray)):
            content = file_bytes
        else:
            content = file_bytes.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            docs = cls.load_from_file(temp_path)
            # Fix metadata to reflect original file name
            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["file_name"] = file_name
                doc.metadata["file_type"] = ext.replace(".", "")
                doc.metadata.pop("file_path", None)
            return docs
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @classmethod
    def load_from_url(cls, url: str) -> List[Document]:
        """Loads and extracts text content from a web URL."""
        try:
            loader = WebBaseLoader(web_path=url)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["file_name"] = url
                doc.metadata["file_type"] = "web"
            return documents
        except Exception as e:
            raise RuntimeError(f"Error scraping URL '{url}': {str(e)}") from e

    @classmethod
    def load_from_text(cls, text: str, source_name: str = "raw_text") -> List[Document]:
        """Creates a Document from raw text input."""
        return [
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "file_name": source_name,
                    "file_type": "text",
                },
            )
        ]
