"""
Conversational RAG Pipeline using modern LangChain (LCEL) and Groq.
Maintains multi-turn conversational memory, history-aware contextual retrieval,
and grounded generation with source citations.
"""

import os
import logging
from operator import itemgetter
from typing import Dict, Any, List, Optional, Iterator

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableBranch,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

from src.memory import MemoryManager
from src.config import RAGConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference context "
    "in the chat history, formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just reformulate it if needed "
    "and otherwise return it as is."
)

QA_SYSTEM_PROMPT = """You are a knowledgeable and precise AI assistant specialized in Retrieval-Augmented Generation (RAG).
Answer the user's question accurately using ONLY the context provided below.

Strict Guidelines:
1. Ground your response in the provided context. If the answer is directly found, state it clearly.
2. If the context does not contain enough information to answer the question, state:
   "Based on the provided documents, I could not find information regarding that."
3. Format your answer with clean Markdown (bullet points, bold highlights, tables, or code snippets when helpful).
4. Reference the document source(s) when answering if relevant.
5. Be concise, direct, and professional.

<context>
{context}
</context>"""


def format_docs(docs: List[Document]) -> str:
    """Formats a list of retrieved documents into a context block with source tags."""
    if not docs:
        return "No relevant context found."
    
    formatted = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source") or doc.metadata.get("file_name", "Unknown Source")
        page = doc.metadata.get("page")
        page_info = f" (Page {page + 1})" if page is not None else ""
        formatted.append(f"--- Document [{idx + 1}] Source: {source}{page_info} ---\n{doc.page_content.strip()}")
    
    return "\n\n".join(formatted)


class ConversationalRAGChain:
    """Production-grade Conversational RAG pipeline built with LangChain LCEL & Groq."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        groq_api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.retriever = retriever
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Pass `groq_api_key` or set `GROQ_API_KEY` environment variable."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_manager = memory_manager or MemoryManager()

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=True,
        )

        # Build LCEL Chains
        self._build_pipeline()

    def _build_pipeline(self):
        """Constructs the history-aware retrieval and grounded QA LCEL pipeline."""
        
        # 1. Query Contextualization Chain (Reformulates follow-up queries)
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        contextualize_chain = contextualize_prompt | self.llm | StrOutputParser()

        # History-aware query resolution:
        # If chat_history exists, reformulate query; otherwise pass input through directly.
        def route_query(input_dict: Dict[str, Any]):
            chat_history = input_dict.get("chat_history", [])
            if not chat_history:
                return input_dict["input"]
            return contextualize_chain

        history_aware_query = RunnableBranch(
            (lambda x: not x.get("chat_history"), itemgetter("input")),
            contextualize_chain,
        )

        # 2. History-Aware Retrieval Step
        # Retrieves docs using the reformulated query
        retrieve_docs = history_aware_query | self.retriever

        # 3. QA Prompt & Generation Chain
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 4. End-to-End Pipeline
        # We preserve retrieved 'context', 'input', and generate 'answer'
        rag_core = (
            RunnableParallel(
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
            )
            .assign(
                answer=(
                    RunnableParallel(
                        {
                            "context": lambda x: format_docs(x["context"]),
                            "input": itemgetter("input"),
                            "chat_history": itemgetter("chat_history"),
                        }
                    )
                    | qa_prompt
                    | self.llm
                    | StrOutputParser()
                )
            )
        )

        # 5. Attach Conversation Memory Management
        self.conversational_chain = RunnableWithMessageHistory(
            rag_core,
            self.memory_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def query(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Executes a conversational query synchronously.
        Returns:
            Dict containing 'answer', 'context' (List[Document]), and 'input'.
        """
        result = self.conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return result

    def stream_query(
        self, user_input: str, session_id: str = "default"
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams response tokens in real-time as they arrive from Groq.
        Yields:
            Dicts with type 'token' (str) and finally type 'sources' (List[Document]).
        """
        # First retrieve context docs for this turn
        session_history = self.memory_manager.get_session_history(session_id)
        messages = session_history.messages

        # Reformulate question if history exists
        if messages:
            contextualize_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            reformulate_chain = contextualize_prompt | self.llm | StrOutputParser()
            search_query = reformulate_chain.invoke(
                {"chat_history": messages, "input": user_input}
            )
        else:
            search_query = user_input

        # Retrieve documents
        docs = self.retriever.invoke(search_query)
        formatted_context = format_docs(docs)

        # Stream QA generation
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        stream_chain = qa_prompt | self.llm | StrOutputParser()

        full_answer = []
        for chunk in stream_chain.stream(
            {
                "context": formatted_context,
                "chat_history": messages,
                "input": user_input,
            }
        ):
            full_answer.append(chunk)
            yield {"type": "token", "content": chunk}

        # Update memory with this turn
        session_history.add_user_message(user_input)
        session_history.add_ai_message("".join(full_answer))

        # Return source documents for citation display
        yield {"type": "sources", "documents": docs}

    def get_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """Returns the conversation history for the given session."""
        return self.memory_manager.get_formatted_history(session_id)

    def clear_history(self, session_id: str = "default") -> None:
        """Clears conversation history for the given session."""
        self.memory_manager.clear_session(session_id)
