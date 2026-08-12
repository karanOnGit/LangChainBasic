"""
⚡ Karan Bhardwaj — Conversational AI & Resume RAG Portal
Built with LangChain, Groq API (LLaMA 3.3 70B), and Multi-VectorStore support.
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
# Page Configuration & Modern Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karan Bhardwaj | Conversational AI RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FFA07A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 1.2rem;
    }
    
    .social-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(255, 107, 107, 0.12);
        color: #ff8e53;
        border: 1px solid rgba(255, 107, 107, 0.3);
        text-decoration: none;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    .social-pill:hover {
        background: rgba(255, 107, 107, 0.25);
        color: #ffffff;
        border-color: #ff6b6b;
    }

    .profile-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .source-card {
        border-left: 3px solid #ff6b6b;
        background-color: rgba(255, 107, 107, 0.05);
        padding: 0.75rem 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
    }

    .prompt-btn {
        margin-bottom: 0.5rem;
    }
    
    .footer-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #718096;
        font-size: 0.88rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# State Management Initialization
# -----------------------------------------------------------------------------
default_config = RAGConfig()
RESUME_PATH = Path(__file__).parent / "sample_data" / "karan_bhardwaj_resume.md"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "karan_rag_session_1"

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "indexed_docs_count" not in st.session_state:
    st.session_state.indexed_docs_count = 0

if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0

if "active_vector_store" not in st.session_state:
    st.session_state.active_vector_store = None

if "auto_indexed" not in st.session_state:
    st.session_state.auto_indexed = False


# Helper function to index documents
def initialize_rag_system(docs, api_key, model_name, store_type, embed_name, c_size, c_overlap, k, temp):
    splitter = TextSplitterManager(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = splitter.split_documents(docs)
    embeddings = get_embedding_model(embed_name)
    persist_dir = (
        default_config.chroma_persist_dir
        if store_type == "chroma"
        else default_config.faiss_persist_dir
    )
    vector_store = VectorStoreManager.create_vector_store(
        store_type=store_type,
        documents=chunks,
        embeddings=embeddings,
        persist_dir=persist_dir,
    )
    retriever = VectorStoreManager.get_retriever(vector_store, k=k)
    memory = MemoryManager(max_history_messages=20)
    rag_chain = ConversationalRAGChain(
        retriever=retriever,
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temp,
        memory_manager=memory,
    )
    st.session_state.rag_chain = rag_chain
    st.session_state.indexed_docs_count = len(docs)
    st.session_state.indexed_chunks_count = len(chunks)
    st.session_state.active_vector_store = store_type
    return rag_chain


# -----------------------------------------------------------------------------
# Sidebar: Profile & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    # Profile Card
    st.markdown(
        """
        <div class="profile-card">
            <h3 style="margin-top:0; margin-bottom:0.3rem; color:#fff;">👨‍💻 Karan Bhardwaj</h3>
            <p style="color:#ff8e53; font-weight:600; font-size:0.9rem; margin-bottom:0.8rem;">
                Full Stack & AI Systems Engineer
            </p>
            <p style="font-size:0.82rem; color:#cbd5e0; line-height:1.4; margin-bottom:1rem;">
                Specialized in architecting scalable automation-driven web applications, Generative AI, LangChain, and Groq API systems.
            </p>
            <div>
                <a class="social-pill" href="https://karanbhardwaj.in" target="_blank">🌐 Portfolio</a>
                <a class="social-pill" href="https://linkedin.com/in/karan-bhardwaj" target="_blank">💼 LinkedIn</a>
                <a class="social-pill" href="https://github.com/karanongit" target="_blank">🐙 GitHub</a>
                <a class="social-pill" href="mailto:karanbhardwaj1107@gmail.com">✉️ Email</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ RAG Engine Settings")

    # API Keys
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Groq API key for ultra-fast LPU inference (auto-loaded from .env)",
    )

    # Model Selection
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=AVAILABLE_GROQ_MODELS,
        index=0,
    )

    # Vector Store Selection
    selected_store = st.selectbox(
        "Vector Store Backend",
        options=SUPPORTED_VECTOR_STORES,
        index=0,
        format_func=lambda s: {
            "faiss": "FAISS (Local In-Memory / Ultra-Fast)",
            "chroma": "ChromaDB (Local Persistent)",
            "pinecone": "Pinecone (Cloud Managed)",
        }.get(s, s.upper()),
    )

    # Advanced Settings Expander
    with st.expander("🔧 Advanced Parameters", expanded=False):
        selected_embedding = st.selectbox(
            "Embedding Model",
            options=AVAILABLE_EMBEDDING_MODELS,
            index=0,
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=800, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150, step=25)
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=3, step=1)

    st.markdown("---")

    # Metrics
    st.markdown("### 📊 Active Knowledge Index")
    col1, col2 = st.columns(2)
    col1.metric("Documents", st.session_state.indexed_docs_count)
    col2.metric("Chunks", st.session_state.indexed_chunks_count)
    if st.session_state.active_vector_store:
        st.caption(f"Backend: `{st.session_state.active_vector_store.upper()}`")

    st.markdown("---")

    # Reset Actions
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain.clear_history(st.session_state.session_id)
        st.toast("Chat memory cleared!", icon="🧹")
        st.rerun()

    # Export Transcript
    if st.session_state.messages:
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="💾 Download Conversation (JSON)",
            data=chat_export,
            file_name="karan_bhardwaj_ai_chat.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Auto-Index Karan Bhardwaj Resume on Initial Load
# -----------------------------------------------------------------------------
if groq_api_key and not st.session_state.auto_indexed and RESUME_PATH.exists():
    try:
        resume_docs = DocumentLoaderManager.load_from_file(str(RESUME_PATH))
        initialize_rag_system(
            docs=resume_docs,
            api_key=groq_api_key,
            model_name=selected_model,
            store_type=selected_store,
            embed_name=AVAILABLE_EMBEDDING_MODELS[0],
            c_size=800,
            c_overlap=150,
            k=3,
            temp=0.2,
        )
        st.session_state.auto_indexed = True
    except Exception as e:
        st.sidebar.error(f"Auto-index error: {e}")


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">⚡ Karan Bhardwaj — Conversational AI & Resume RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask anything about <strong>Karan Bhardwaj</strong>\'s professional experience, projects, skills, and AI engineering expertise. Powered by <strong>LangChain</strong> and <strong>Groq LPU (LLaMA 3.3 70B)</strong>.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Quick Starter Prompts
# -----------------------------------------------------------------------------
st.markdown("##### 💡 Suggested Questions")
q_col1, q_col2, q_col3, q_col4 = st.columns(4)

selected_starter_prompt = None

with q_col1:
    if st.button("🛠️ Technical Skills", use_container_width=True):
        selected_starter_prompt = "What are Karan Bhardwaj's core technical skills, programming languages, and AI expertise?"

with q_col2:
    if st.button("🚀 Projects Overview", use_container_width=True):
        selected_starter_prompt = "Tell me about Karan's key projects like FHMNews and Socioglamm."

with q_col3:
    if st.button("💼 Work Experience", use_container_width=True):
        selected_starter_prompt = "What is Karan's work experience at Creative Volt and Flyhead Media?"

with q_col4:
    if st.button("📬 Contact & Links", use_container_width=True):
        selected_starter_prompt = "How can I contact Karan Bhardwaj, and what are his portfolio, GitHub, and LinkedIn links?"


# -----------------------------------------------------------------------------
# Document Ingestion / Custom Document Expander
# -----------------------------------------------------------------------------
with st.expander("📂 Knowledge Source & Document Manager", expanded=not st.session_state.auto_indexed):
    tab_current, tab_custom = st.tabs(["📄 Active Resume Knowledge Base", "➕ Upload Additional Document"])

    with tab_current:
        st.markdown(
            f"""
            **Currently Loaded Document**: `karan_bhardwaj_resume.md`  
            - **Candidate**: **Karan Bhardwaj** (Full Stack & AI Engineer)
            - **Indexed Status**: {'✅ Active and ready to chat' if st.session_state.rag_chain else '⚠️ Needs API key'}
            - **Contact**: `karanbhardwaj1107@gmail.com` | `+91 6202640773`
            - **Portfolio**: [karanbhardwaj.in](https://karanbhardwaj.in)
            """
        )
        if st.button("🔄 Re-Index Karan's Resume", type="primary"):
            if not groq_api_key:
                st.error("Please enter your Groq API Key in the sidebar.")
            elif RESUME_PATH.exists():
                with st.spinner("Indexing Karan Bhardwaj's Resume..."):
                    resume_docs = DocumentLoaderManager.load_from_file(str(RESUME_PATH))
                    initialize_rag_system(
                        docs=resume_docs,
                        api_key=groq_api_key,
                        model_name=selected_model,
                        store_type=selected_store,
                        embed_name=selected_embedding,
                        c_size=chunk_size,
                        c_overlap=chunk_overlap,
                        k=top_k,
                        temp=temperature,
                    )
                    st.success("Successfully indexed Karan Bhardwaj's resume!")
                    st.rerun()

    with tab_custom:
        custom_files = st.file_uploader(
            "Upload an additional document (PDF, TXT, DOCX, Markdown):",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
        )
        if custom_files and st.button("Index Uploaded Documents"):
            if not groq_api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                with st.spinner("Processing documents..."):
                    docs = []
                    for f in custom_files:
                        docs.extend(DocumentLoaderManager.load_from_upload(f.name, f.read()))
                    initialize_rag_system(
                        docs=docs,
                        api_key=groq_api_key,
                        model_name=selected_model,
                        store_type=selected_store,
                        embed_name=selected_embedding,
                        c_size=chunk_size,
                        c_overlap=chunk_overlap,
                        k=top_k,
                        temp=temperature,
                    )
                    st.success(f"Indexed {len(docs)} custom document sections!")
                    st.rerun()


# -----------------------------------------------------------------------------
# Conversational Chat Area
# -----------------------------------------------------------------------------
st.markdown("---")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources & Resume Citations", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("source", "Karan Bhardwaj Resume")
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>[{idx}] {source_name}</strong><br>
                            <span style="font-size:0.88rem; color:#cbd5e0;">{src.get('content')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Determine user input (from chat box or quick prompt buttons)
user_query = st.chat_input("Ask a question about Karan Bhardwaj...")
active_prompt = selected_starter_prompt or user_query

if active_prompt:
    if not groq_api_key:
        st.error("Please enter a Groq API Key in the sidebar to chat.")
    elif not st.session_state.rag_chain:
        if RESUME_PATH.exists():
            with st.spinner("Initializing RAG index..."):
                resume_docs = DocumentLoaderManager.load_from_file(str(RESUME_PATH))
                initialize_rag_system(
                    docs=resume_docs,
                    api_key=groq_api_key,
                    model_name=selected_model,
                    store_type=selected_store,
                    embed_name=selected_embedding,
                    c_size=chunk_size,
                    c_overlap=chunk_overlap,
                    k=top_k,
                    temp=temperature,
                )
                st.session_state.auto_indexed = True

    if st.session_state.rag_chain and groq_api_key:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Stream response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_reply = ""
            retrieved_sources = []

            try:
                for chunk in st.session_state.rag_chain.stream_query(
                    active_prompt, session_id=st.session_state.session_id
                ):
                    if chunk["type"] == "token":
                        full_reply += chunk["content"]
                        placeholder.markdown(full_reply + "▌")
                    elif chunk["type"] == "sources":
                        retrieved_sources = chunk["documents"]

                placeholder.markdown(full_reply)

                # Render sources
                formatted_sources = []
                if retrieved_sources:
                    with st.expander("🔍 View Retrieved Sources & Resume Citations", expanded=False):
                        for idx, doc in enumerate(retrieved_sources, 1):
                            source_title = doc.metadata.get("source") or "Karan Bhardwaj Resume"
                            src_info = {
                                "source": source_title,
                                "content": doc.page_content.strip(),
                            }
                            formatted_sources.append(src_info)
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <strong>[{idx}] {source_title}</strong><br>
                                    <span style="font-size:0.88rem; color:#cbd5e0;">{src_info['content']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_reply,
                        "sources": formatted_sources,
                    }
                )

            except Exception as e:
                st.error(f"Error during response generation: {str(e)}")


# -----------------------------------------------------------------------------
# Footer Signature
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-container">
        ⚡ <strong>Karan Bhardwaj</strong> | Full Stack & AI Systems Engineer<br>
        <a href="https://karanbhardwaj.in" target="_blank" style="color:#ff8e53; text-decoration:none; margin:0 8px;">Portfolio</a> •
        <a href="https://linkedin.com/in/karan-bhardwaj" target="_blank" style="color:#ff8e53; text-decoration:none; margin:0 8px;">LinkedIn</a> •
        <a href="https://github.com/karanongit" target="_blank" style="color:#ff8e53; text-decoration:none; margin:0 8px;">GitHub</a> •
        <a href="mailto:karanbhardwaj1107@gmail.com" style="color:#ff8e53; text-decoration:none; margin:0 8px;">Contact</a><br>
        <span style="font-size:0.78rem; color:#4a5568;">Crafted with LangChain & Groq LPU™ (LLaMA 3.3 70B)</span>
    </div>
    """,
    unsafe_allow_html=True,
)
