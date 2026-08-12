"""
⚡ Karan Bhardwaj — Conversational AI & Resume RAG Portal
Engineered with LangChain LCEL, Groq LPU™ (LLaMA 3.3 70B), and Multi-VectorStore Architecture.
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
# 1. Page Configuration & Professional Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karan Bhardwaj | AI Systems & Resume RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-dark: #0B0F19;
        --card-bg: #151C2C;
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        --accent-blue: #3B82F6;
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Cleanup */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1100px !important;
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    /* Sidebar Profile Card */
    .profile-container {
        background: linear-gradient(180deg, #182235 0%, #111726 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .profile-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-gradient);
    }

    .avatar-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        border-radius: 16px;
        background: var(--accent-gradient);
        color: white;
        font-weight: 800;
        font-size: 1.4rem;
        box-shadow: 0 8px 16px -4px rgba(99, 102, 241, 0.5);
        margin-bottom: 0.85rem;
    }

    .profile-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .verified-icon {
        color: #38BDF8;
        font-size: 1rem;
    }

    .profile-role {
        font-size: 0.85rem;
        font-weight: 600;
        color: #818CF8;
        margin-top: 2px;
        margin-bottom: 0.75rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        background: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.25);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 8px #22C55E;
    }

    .social-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    .social-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.45rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #E2E8F0 !important;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }

    .social-btn:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }

    /* Hero Header */
    .hero-wrapper {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.75rem;
        position: relative;
        backdrop-filter: blur(16px);
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 9999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }

    .hero-title span {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #94A3B8;
        font-size: 0.98rem;
        line-height: 1.6;
        max-width: 850px;
        margin: 0;
    }

    /* Quick Prompt Cards */
    .card-prompt-btn {
        background: #151C2C;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
        margin-bottom: 0.75rem;
    }

    .card-prompt-btn:hover {
        border-color: #6366F1;
        background: #1A2338;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px -5px rgba(99, 102, 241, 0.25);
    }

    /* Source Citation Cards */
    .source-box {
        background: #111726;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-left: 3px solid #818CF8;
        border-radius: 0 12px 12px 0;
        padding: 0.85rem 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .source-title {
        color: #A5B4FC;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .source-snippet {
        color: #CBD5E1;
        font-size: 0.85rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Professional Footer */
    .site-footer {
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
        color: #64748B;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 3.5rem;
    }
    .site-footer a {
        color: #818CF8;
        text-decoration: none;
        margin: 0 6px;
        font-weight: 500;
    }
    .site-footer a:hover {
        color: #C7D2FE;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. State & Knowledge Ingestion Engine
# -----------------------------------------------------------------------------
RESUME_PATH = Path(__file__).parent / "sample_data" / "karan_bhardwaj_resume.md"
default_config = RAGConfig()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "karan_ai_session"

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0

if "active_vector_store" not in st.session_state:
    st.session_state.active_vector_store = "FAISS"

if "quick_prompt_query" not in st.session_state:
    st.session_state.quick_prompt_query = None


# Cached RAG Initializer to ensure instantaneous response times
@st.cache_resource(show_spinner=False)
def load_cached_rag_engine(api_key: str, model_name: str, store_type: str, embedding_model: str):
    """Builds and caches the vector store & RAG pipeline from Karan Bhardwaj's resume."""
    if not RESUME_PATH.exists():
        return None, 0

    docs = DocumentLoaderManager.load_from_file(str(RESUME_PATH))
    splitter = TextSplitterManager(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model(embedding_model)
    persist_dir = (
        default_config.chroma_persist_dir
        if store_type.lower() == "chroma"
        else default_config.faiss_persist_dir
    )

    vector_store = VectorStoreManager.create_vector_store(
        store_type=store_type,
        documents=chunks,
        embeddings=embeddings,
        persist_dir=persist_dir,
    )
    retriever = VectorStoreManager.get_retriever(vector_store, k=3)
    memory = MemoryManager(max_history_messages=20)

    chain = ConversationalRAGChain(
        retriever=retriever,
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        memory_manager=memory,
    )
    return chain, len(chunks)


# -----------------------------------------------------------------------------
# 3. Sidebar: Developer Profile & System Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    # Developer Profile Card
    st.markdown(
        """
        <div class="profile-container">
            <div class="avatar-badge">KB</div>
            <h2 class="profile-name">
                Karan Bhardwaj <span class="verified-icon">✦</span>
            </h2>
            <div class="profile-role">Full Stack & AI Systems Engineer</div>
            <div class="status-pill">
                <div class="status-dot"></div> Available for AI & Full-Stack Roles
            </div>
            <div style="font-size:0.8rem; color:#94A3B8; line-height:1.45; margin-bottom:1rem;">
                📍 Sahibzada Ajit Singh Nagar, Punjab, India<br>
                🎓 B.Tech CSE — Galgotias University
            </div>
            <div class="social-grid">
                <a class="social-btn" href="https://karanbhardwaj.in" target="_blank">🌐 Portfolio</a>
                <a class="social-btn" href="https://linkedin.com/in/karan-bhardwaj" target="_blank">💼 LinkedIn</a>
                <a class="social-btn" href="https://github.com/karanongit" target="_blank">🐙 GitHub</a>
                <a class="social-btn" href="mailto:karanbhardwaj1107@gmail.com">✉️ Email</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚡ Inference & RAG Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Groq API key (auto-loaded from .env). Powers ultra-low-latency LPU inference.",
    )

    selected_model = st.selectbox(
        "Groq LLM Model",
        options=AVAILABLE_GROQ_MODELS,
        index=0,
        help="LLaMA 3.3 70B delivers state-of-the-art conversational reasoning at >300 tokens/sec.",
    )

    selected_store = st.selectbox(
        "Vector Store Backend",
        options=SUPPORTED_VECTOR_STORES,
        index=0,
        format_func=lambda s: {
            "faiss": "FAISS (In-Memory / Instant Search)",
            "chroma": "ChromaDB (Persistent Vector DB)",
            "pinecone": "Pinecone (Cloud Managed)",
        }.get(s, s.upper()),
    )

    with st.expander("⚙️ Advanced Parameters", expanded=False):
        selected_embedding = st.selectbox(
            "Embedding Model",
            options=AVAILABLE_EMBEDDING_MODELS,
            index=0,
        )
        chunk_size = st.slider("Chunk Size", 200, 2000, 800, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 150, 25)
        top_k = st.slider("Top-K Retrieval", 1, 8, 3, 1)

    st.markdown("---")

    # System Status Indicator
    st.markdown("### 📊 Active Knowledge Status")
    k_col1, k_col2 = st.columns(2)
    k_col1.metric("Documents", "1 (Resume)")
    k_col2.metric("Index Chunks", st.session_state.indexed_chunks_count or 3)

    st.caption(f"Engine: `Groq LPU` | Model: `{selected_model.split('-')[0].upper()}` | VectorStore: `{selected_store.upper()}`")

    st.markdown("---")

    # Actions
    if st.button("🧹 Clear Conversation Memory", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain.clear_history(st.session_state.session_id)
        st.toast("Chat memory reset!", icon="🧹")
        st.rerun()

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
# 4. Auto-Initialize RAG Pipeline
# -----------------------------------------------------------------------------
if groq_api_key:
    try:
        chain, count = load_cached_rag_engine(
            api_key=groq_api_key,
            model_name=selected_model,
            store_type=selected_store,
            embedding_model=AVAILABLE_EMBEDDING_MODELS[0],
        )
        st.session_state.rag_chain = chain
        st.session_state.indexed_chunks_count = count
        st.session_state.active_vector_store = selected_store
    except Exception as e:
        st.sidebar.error(f"Initialization notice: {e}")


# -----------------------------------------------------------------------------
# 5. Main Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-tag">⚡ Live Conversational AI Persona</div>
        <h1 class="hero-title">Karan Bhardwaj <span>AI Assistant</span></h1>
        <p class="hero-desc">
            Explore <strong>Karan Bhardwaj's</strong> professional journey, engineering achievements, autonomous AI systems (such as <em>FHMNews.com</em> &amp; <em>Socioglamm</em>), full-stack architectures, and technical skills in real time. Powered by <strong>LangChain LCEL</strong> and <strong>Groq LPU™ (LLaMA 3.3 70B)</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 6. Interactive Quick-Prompt Chips
# -----------------------------------------------------------------------------
st.markdown("#### 💡 Suggested Inquiries")
col_q1, col_q2 = st.columns(2)
col_q3, col_q4 = st.columns(2)

with col_q1:
    if st.button("🛠️ Technical Skills & AI Stack", use_container_width=True, help="Query Karan's frontend, backend, and AI proficiencies"):
        st.session_state.quick_prompt_query = "What are Karan Bhardwaj's core technical skills, programming languages, and AI proficiencies?"

with col_q2:
    if st.button("🚀 Featured Projects (FHMNews & Socioglamm)", use_container_width=True, help="Query details on Karan's key project architectures"):
        st.session_state.quick_prompt_query = "Tell me in detail about Karan's key projects such as FHMNews, Socioglamm, and Carsnbike."

with col_q3:
    if st.button("💼 Work Experience & Roles", use_container_width=True, help="Query experience at Creative Volt and Flyhead Media"):
        st.session_state.quick_prompt_query = "Summarize Karan Bhardwaj's work experience at Creative Volt and Flyhead Media."

with col_q4:
    if st.button("📬 Education, Portfolio & Contact", use_container_width=True, help="Query degree, links, and direct contact info"):
        st.session_state.quick_prompt_query = "What is Karan Bhardwaj's education background, and how can I contact him directly?"

st.markdown("---")


# -----------------------------------------------------------------------------
# 7. Conversational Chat Display
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👨‍💻" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Verified Resume Citations", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-title">📄 Citation [{idx}] — {src.get('source', 'Karan Bhardwaj Resume')}</div>
                            <div class="source-snippet">{src.get('content')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Process Prompt (from input box or quick prompt click)
typed_input = st.chat_input("Ask a question about Karan Bhardwaj...")
prompt_to_run = st.session_state.quick_prompt_query or typed_input

# Reset quick prompt state after reading
if st.session_state.quick_prompt_query:
    st.session_state.quick_prompt_query = None

if prompt_to_run:
    if not groq_api_key:
        st.error("Please enter a valid Groq API Key in the sidebar to enable the AI Persona.")
    elif not st.session_state.rag_chain:
        st.warning("RAG engine is initializing. Please click Re-Index in the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(prompt_to_run)

        # Stream assistant response
        with st.chat_message("assistant", avatar="⚡"):
            response_container = st.empty()
            streaming_text = ""
            retrieved_sources = []

            try:
                for chunk in st.session_state.rag_chain.stream_query(
                    prompt_to_run, session_id=st.session_state.session_id
                ):
                    if chunk["type"] == "token":
                        streaming_text += chunk["content"]
                        response_container.markdown(streaming_text + " ▌")
                    elif chunk["type"] == "sources":
                        retrieved_sources = chunk["documents"]

                # Render final markdown without cursor
                response_container.markdown(streaming_text)

                # Format source citations
                formatted_sources = []
                if retrieved_sources:
                    with st.expander("🔍 Verified Resume Citations", expanded=False):
                        for idx, doc in enumerate(retrieved_sources, 1):
                            src_name = doc.metadata.get("source", "Karan Bhardwaj Resume")
                            content = doc.page_content.strip()
                            formatted_sources.append({"source": src_name, "content": content})
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <div class="source-title">📄 Citation [{idx}] — {src_name}</div>
                                    <div class="source-snippet">{content}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save assistant turn to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": streaming_text,
                        "sources": formatted_sources,
                    }
                )

            except Exception as e:
                st.error(f"Error during response synthesis: {str(e)}")


# -----------------------------------------------------------------------------
# 8. Modern Footer Signature
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="site-footer">
        ⚡ <strong>Karan Bhardwaj</strong> — Full Stack & AI Systems Engineer<br>
        <a href="https://karanbhardwaj.in" target="_blank">Portfolio</a> •
        <a href="https://linkedin.com/in/karan-bhardwaj" target="_blank">LinkedIn</a> •
        <a href="https://github.com/karanongit" target="_blank">GitHub</a> •
        <a href="mailto:karanbhardwaj1107@gmail.com">karanbhardwaj1107@gmail.com</a> •
        <a href="tel:+916202640773">+91 6202640773</a><br>
        <span style="font-size:0.75rem; color:#475569; margin-top:0.35rem; display:inline-block;">
            Engineered with LangChain LCEL &amp; Groq LPU™ (LLaMA 3.3 70B)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
