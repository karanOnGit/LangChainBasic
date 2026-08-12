"""
Text Splitting and Chunking Module.
Uses RecursiveCharacterTextSplitter with metadata enrichment.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitterManager:
    """Manager for chunking documents into optimal sizes for vector storage and retrieval."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents and enriches chunks with index metadata."""
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Add chunk-specific metadata
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["chunk_id"] = f"{source}#chunk_{idx}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def split_text(self, text: str, source: str = "raw_text") -> List[Document]:
        """Splits plain text into structured Document chunks."""
        doc = Document(page_content=text, metadata={"source": source})
        return self.split_documents([doc])
