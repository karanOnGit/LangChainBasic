"""
Streamlit Web Application for Conversational RAG with LangChain and Groq.
Features multi-turn conversation memory, dynamic document loading, multi-vectorstore switching,
real-time streaming responses, and interactive source attribution.
"""

import os
import json
from pathlib import Path
import streamlit as st

from src.config import (
    RAGConfig,
    AVAILABLE_GROQ_MODELS,
    AVAILABLE_EMBEDDING_MODELS,
    SUPPORTED_VECTOR_STORES,
)
from src.loaders import DocumentLoaderManager
from src.splitters import TextSplitterManager
from src.embeddings import get_embedding_model
from src.vectorstores import VectorStoreManager
from src.memory import MemoryManager
from src.rag_chain import ConversationalRAGChain

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Conversational RAG | LangChain + Groq",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f5576c 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #2e3440;
        color: #eceff4;
        margin-right: 0.4rem;
    }
    .source-card {
        border-left: 3px solid #f5576c;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# State Management Initialization
# -----------------------------------------------------------------------------
default_config = RAGConfig()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session_1"

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "indexed_docs_count" not in st.session_state:
    st.session_state.indexed_docs_count = 0

if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0

if "active_vector_store" not in st.session_state:
    st.session_state.active_vector_store = None


# -----------------------------------------------------------------------------
# Sidebar: Settings & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Keys
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free high-speed API key at https://console.groq.com",
    )

    # Model Selection
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=AVAILABLE_GROQ_MODELS,
        index=0,
        help="Choose ultra-low latency models powered by Groq LPUs.",
    )

    # Vector Store Selection
    selected_store = st.selectbox(
        "Vector Store Backend",
        options=SUPPORTED_VECTOR_STORES,
        index=0,
        format_func=lambda s: {
            "faiss": "FAISS (Local In-Memory / Fast)",
            "chroma": "ChromaDB (Local Persistent)",
            "pinecone": "Pinecone (Cloud Managed)",
        }.get(s, s.upper()),
    )

    # Pinecone specific options
    pinecone_key = ""
    pinecone_index = ""
    if selected_store == "pinecone":
        pinecone_key = st.text_input(
            "Pinecone API Key",
            value=os.getenv("PINECONE_API_KEY", ""),
            type="password",
        )
        pinecone_index = st.text_input(
            "Pinecone Index Name",
            value=os.getenv("PINECONE_INDEX_NAME", "langchain-rag-index"),
        )

    # Advanced Settings Expander
    with st.expander("🔧 Advanced RAG Parameters", expanded=False):
        selected_embedding = st.selectbox(
            "Embedding Model",
            options=AVAILABLE_EMBEDDING_MODELS,
            index=0,
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=25)
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=4, step=1)

    st.markdown("---")

    # Knowledge Base Stats
    st.markdown("### 📊 Index Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Documents", st.session_state.indexed_docs_count)
    col2.metric("Chunks", st.session_state.indexed_chunks_count)
    if st.session_state.active_vector_store:
        st.markdown(f"**Active Store:** `{st.session_state.active_vector_store.upper()}`")

    st.markdown("---")

    # Memory & Session Actions
    if st.button("🗑️ Clear Conversation Memory", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain.clear_history(st.session_state.session_id)
        st.toast("Chat memory cleared!", icon="🧹")
        st.rerun()

    # Export Chat
    if st.session_state.messages:
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="💾 Export Chat Transcript (JSON)",
            data=chat_export,
            file_name="rag_chat_transcript.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Conversational RAG with LangChain & Groq</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-turn history-aware Q&A across multi-format documents with pluggable Vector Stores.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Document Ingestion Section
# -----------------------------------------------------------------------------
with st.expander("📂 Ingest & Index Knowledge Sources", expanded=st.session_state.indexed_chunks_count == 0):
    tab1, tab2, tab3 = st.tabs(["📁 Upload Files", "🌐 Scrape Web URL", "💡 Load Sample Dataset"])

    loaded_documents = []

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX, Markdown)",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for file in uploaded_files:
                docs = DocumentLoaderManager.load_from_upload(file.name, file.read())
                loaded_documents.extend(docs)
            st.info(f"Loaded {len(loaded_documents)} document sections from {len(uploaded_files)} uploaded file(s).")

    with tab2:
        web_url = st.text_input("Enter Web Page URL to scrape:", placeholder="https://example.com/article")
        if st.button("Fetch URL Content"):
            if web_url:
                with st.spinner("Scraping webpage..."):
                    try:
                        url_docs = DocumentLoaderManager.load_from_url(web_url)
                        loaded_documents.extend(url_docs)
                        st.success(f"Successfully scraped content from {web_url}")
                    except Exception as e:
                        st.error(f"Failed to scrape URL: {str(e)}")

    with tab3:
        st.markdown("Choose a pre-packaged sample dataset for instant testing:")
        sample_choice = st.radio(
            "Sample Dataset",
            options=["AI & Autonomous Agents Guide (2026)", "Acme Corp Remote Work & Security Policies"],
            index=0,
        )
        if st.button("Load Selected Sample Data"):
            sample_dir = Path(__file__).parent / "sample_data"
            sample_file = (
                sample_dir / "knowledge_base.md"
                if "Autonomous" in sample_choice
                else sample_dir / "company_policy.txt"
            )
            if sample_file.exists():
                sample_docs = DocumentLoaderManager.load_from_file(str(sample_file))
                loaded_documents.extend(sample_docs)
                st.success(f"Loaded sample file: `{sample_file.name}` ({len(sample_docs)} document sections)")

    st.markdown("---")

    # Ingestion Action Button
    if st.button("🚀 Index & Build Vector Store", type="primary", use_container_width=True):
        if not groq_api_key:
            st.error("Please provide a valid Groq API Key in the sidebar.")
        elif not loaded_documents:
            st.warning("Please upload files, enter a URL, or load a sample dataset first.")
        else:
            with st.status("Indexing documents and building vector store...", expanded=True) as status:
                st.write("1. Splitting documents into semantic chunks...")
                splitter = TextSplitterManager(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.split_documents(loaded_documents)
                st.write(f"✓ Generated {len(chunks)} text chunks.")

                st.write(f"2. Initializing embedding model `{selected_embedding}`...")
                embeddings = get_embedding_model(selected_embedding)

                st.write(f"3. Constructing `{selected_store.upper()}` Vector Store...")
                persist_dir = (
                    default_config.chroma_persist_dir
                    if selected_store == "chroma"
                    else default_config.faiss_persist_dir
                )

                vector_store = VectorStoreManager.create_vector_store(
                    store_type=selected_store,
                    documents=chunks,
                    embeddings=embeddings,
                    persist_dir=persist_dir,
                    pinecone_api_key=pinecone_key if selected_store == "pinecone" else None,
                    pinecone_index_name=pinecone_index if selected_store == "pinecone" else None,
                )

                retriever = VectorStoreManager.get_retriever(vector_store, k=top_k)

                st.write("4. Initializing Conversational RAG Engine with Memory...")
                memory = MemoryManager(max_history_messages=20)
                st.session_state.rag_chain = ConversationalRAGChain(
                    retriever=retriever,
                    groq_api_key=groq_api_key,
                    model_name=selected_model,
                    temperature=temperature,
                    memory_manager=memory,
                )

                st.session_state.indexed_docs_count = len(loaded_documents)
                st.session_state.indexed_chunks_count = len(chunks)
                st.session_state.active_vector_store = selected_store

                status.update(
                    label=f"✓ Ingestion Complete! {len(chunks)} chunks indexed into {selected_store.upper()}",
                    state="complete",
                    expanded=False,
                )
                st.rerun()


# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
st.markdown("### 💬 Conversational Dialogue")

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("source", "Unknown")
                    page_info = f" | Page {src.get('page') + 1}" if src.get("page") is not None else ""
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>[{idx}] {source_name}{page_info}</strong><br>
                            <span style="font-size:0.9rem; color:#bbb;">{src.get('content')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Handle New User Input
if user_prompt := st.chat_input("Ask a question about your indexed documents..."):
    if not st.session_state.rag_chain:
        st.warning("Please index documents above or load sample data before asking questions.")
    elif not groq_api_key:
        st.error("Please enter a Groq API Key in the sidebar.")
    else:
        # Append and render user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            retrieved_sources = []

            try:
                for chunk in st.session_state.rag_chain.stream_query(
                    user_prompt, session_id=st.session_state.session_id
                ):
                    if chunk["type"] == "token":
                        full_response += chunk["content"]
                        response_placeholder.markdown(full_response + "▌")
                    elif chunk["type"] == "sources":
                        retrieved_sources = chunk["documents"]

                # Render final full response without cursor
                response_placeholder.markdown(full_response)

                # Format source citations
                formatted_sources = []
                if retrieved_sources:
                    with st.expander("🔍 View Retrieved Sources", expanded=False):
                        for idx, doc in enumerate(retrieved_sources, 1):
                            src_meta = {
                                "source": doc.metadata.get("source")
                                or doc.metadata.get("file_name", "Unknown"),
                                "page": doc.metadata.get("page"),
                                "content": doc.page_content.strip(),
                            }
                            formatted_sources.append(src_meta)

                            page_info = (
                                f" | Page {src_meta['page'] + 1}"
                                if src_meta["page"] is not None
                                else ""
                            )
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <strong>[{idx}] {src_meta['source']}{page_info}</strong><br>
                                    <span style="font-size:0.9rem; color:#bbb;">{src_meta['content']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save assistant message to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "sources": formatted_sources,
                    }
                )

            except Exception as e:
                st.error(f"Error during RAG execution: {str(e)}")
