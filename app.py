"""
⚡ Karan Bhardwaj — Conversational AI & Resume RAG Portal
Minimalist Light Theme Edition | Built with LangChain LCEL & Groq LPU™ (LLaMA 3.3 70B)
"""

import os
import json
import base64
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
# 1. Page Configuration & Minimalist Light Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karan Bhardwaj | Conversational AI RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESUME_PATH = Path(__file__).parent / "sample_data" / "karan_bhardwaj_resume.md"
IMAGE_PATH = Path(__file__).parent / "assets" / "karan_bhardwaj.png"
default_config = RAGConfig()


def get_base64_image(image_path: Path) -> str:
    """Returns base64 string for embedding images directly in HTML."""
    if image_path.exists():
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return ""


profile_img_b64 = get_base64_image(IMAGE_PATH)
img_src_html = f"data:image/png;base64,{profile_img_b64}" if profile_img_b64 else ""

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg-main: #FAFAFA;
        --card-bg: #FFFFFF;
        --border-color: #E2E8F0;
        --border-hover: #CBD5E1;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --accent-primary: #0F172A;
        --accent-blue: #2563EB;
        --accent-pill: #F1F5F9;
    }}

    * {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Global Streamlit Elements */
    .stApp {{
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }}

    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 980px !important;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background: transparent !important;}}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-color) !important;
    }}

    /* Profile Card */
    .profile-card {{
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        text-align: center;
    }}

    .avatar-img {{
        width: 84px;
        height: 84px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 0.75rem auto;
        display: block;
        border: 2px solid #F1F5F9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }}

    .profile-name {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.2rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
    }}

    .profile-role {{
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 0.65rem;
    }}

    .avail-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 500;
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.85rem;
    }}

    .avail-dot {{
        width: 6px;
        height: 6px;
        background: #22C55E;
        border-radius: 50%;
    }}

    .social-btn-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }}

    .social-link {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #F8FAFC;
        border: 1px solid var(--border-color);
        color: var(--text-primary) !important;
        text-decoration: none !important;
        padding: 0.4rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.15s ease;
    }}

    .social-link:hover {{
        background: #F1F5F9;
        border-color: var(--border-hover);
        color: #000000 !important;
    }}

    /* Hero Header */
    .hero-header {{
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }}

    .hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        background: #F1F5F9;
        color: var(--text-secondary);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }}

    .hero-title {{
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.015em;
    }}

    .hero-desc {{
        font-size: 0.92rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0;
    }}

    /* Prompt Cards */
    .prompt-chip {{
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        font-size: 0.85rem;
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.15s ease;
        text-align: left;
        cursor: pointer;
    }}

    .prompt-chip:hover {{
        border-color: #0F172A;
        background: #F8FAFC;
    }}

    /* Source Cards */
    .citation-card {{
        background: #F8FAFC;
        border: 1px solid var(--border-color);
        border-left: 3px solid #0F172A;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 0.9rem;
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
    }}

    .citation-title {{
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }}

    .citation-text {{
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}

    /* Minimal Footer */
    .minimal-footer {{
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        color: var(--text-muted);
        font-size: 0.82rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }}
    .minimal-footer a {{
        color: var(--text-primary);
        text-decoration: none;
        margin: 0 6px;
        font-weight: 500;
    }}
    .minimal-footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Session State & Cached RAG Engine
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "karan_minimal_session"

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0

if "active_vector_store" not in st.session_state:
    st.session_state.active_vector_store = "FAISS"

if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = None


@st.cache_resource(show_spinner=False)
def load_rag_pipeline(api_key: str, model_name: str, store_type: str):
    """Initializes the vector store and LCEL Conversational RAG pipeline."""
    if not RESUME_PATH.exists():
        return None, 0

    docs = DocumentLoaderManager.load_from_file(str(RESUME_PATH))
    splitter = TextSplitterManager(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
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
# 3. Sidebar: Profile & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    # Minimal Profile Card
    avatar_html = (
        f'<img src="{img_src_html}" class="avatar-img" alt="Karan Bhardwaj" />'
        if img_src_html
        else '<div style="width:76px; height:76px; background:#0F172A; color:#FFF; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; margin:0 auto 0.75rem auto;">KB</div>'
    )

    st.markdown(
        f"""
        <div class="profile-card">
            {avatar_html}
            <h2 class="profile-name">Karan Bhardwaj</h2>
            <div class="profile-role">Full Stack &amp; AI Systems Engineer</div>
            <div class="avail-badge">
                <div class="avail-dot"></div> Available for AI &amp; Full Stack Roles
            </div>
            <div style="font-size:0.78rem; color:#64748B; margin-bottom:0.75rem;">
                📍 Sahibzada Ajit Singh Nagar, Punjab<br>
                🎓 B.Tech CSE — Galgotias University
            </div>
            <div class="social-btn-grid">
                <a class="social-link" href="https://karanbhardwaj.in" target="_blank">🌐 Portfolio</a>
                <a class="social-link" href="https://linkedin.com/in/karan-bhardwaj" target="_blank">💼 LinkedIn</a>
                <a class="social-link" href="https://github.com/karanongit" target="_blank">🐙 GitHub</a>
                <a class="social-link" href="mailto:karanbhardwaj1107@gmail.com">✉️ Email</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### System Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Groq API Key (auto-loaded from .env). Powers ultra-fast LPU inference.",
    )

    selected_model = st.selectbox(
        "Groq LLM Model",
        options=AVAILABLE_GROQ_MODELS,
        index=0,
    )

    selected_store = st.selectbox(
        "Vector Store Backend",
        options=SUPPORTED_VECTOR_STORES,
        index=0,
        format_func=lambda s: {
            "faiss": "FAISS (In-Memory / Fast)",
            "chroma": "ChromaDB (Persistent)",
            "pinecone": "Pinecone (Cloud)",
        }.get(s, s.upper()),
    )

    st.markdown("---")

    # Knowledge Status
    col1, col2 = st.columns(2)
    col1.metric("Documents", "1 (Resume)")
    col2.metric("Chunks", st.session_state.indexed_chunks_count or 3)

    st.caption(f"Status: `Active` • Engine: `Groq LPU` • VectorStore: `{selected_store.upper()}`")

    st.markdown("---")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain.clear_history(st.session_state.session_id)
        st.toast("Chat reset!", icon="🧹")
        st.rerun()

    if st.session_state.messages:
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="💾 Export Transcript (JSON)",
            data=chat_export,
            file_name="karan_bhardwaj_ai_chat.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# 4. Auto-Initialize Knowledge Base
# -----------------------------------------------------------------------------
if groq_api_key:
    try:
        chain, count = load_rag_pipeline(
            api_key=groq_api_key,
            model_name=selected_model,
            store_type=selected_store,
        )
        st.session_state.rag_chain = chain
        st.session_state.indexed_chunks_count = count
        st.session_state.active_vector_store = selected_store
    except Exception as e:
        st.sidebar.error(f"Initialization notice: {e}")


# -----------------------------------------------------------------------------
# 5. Main Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-pill">⚡ Conversational AI Persona</div>
        <h1 class="hero-title">Karan Bhardwaj</h1>
        <p class="hero-desc">
            Explore <strong>Karan Bhardwaj's</strong> professional experience, autonomous AI pipelines (<em>FHMNews.com</em>, <em>Socioglamm</em>), full-stack web applications, and technical skills in real time. Powered by <strong>LangChain LCEL</strong> and <strong>Groq LPU™ (LLaMA 3.3 70B)</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 6. Minimal Quick-Inquiry Chips
# -----------------------------------------------------------------------------
st.markdown("##### Suggested Inquiries")
q_col1, q_col2 = st.columns(2)
q_col3, q_col4 = st.columns(2)

with q_col1:
    if st.button("🛠️ Technical Skills & AI Stack", use_container_width=True):
        st.session_state.prompt_query = "What are Karan Bhardwaj's core technical skills, programming languages, and AI proficiencies?"

with q_col2:
    if st.button("🚀 Projects (FHMNews, Socioglamm)", use_container_width=True):
        st.session_state.prompt_query = "Tell me in detail about Karan's key projects such as FHMNews, Socioglamm, and Carsnbike."

with q_col3:
    if st.button("💼 Work Experience & Roles", use_container_width=True):
        st.session_state.prompt_query = "Summarize Karan Bhardwaj's work experience at Creative Volt and Flyhead Media."

with q_col4:
    if st.button("📬 Education & Contact Details", use_container_width=True):
        st.session_state.prompt_query = "What is Karan Bhardwaj's education background, and how can I contact him directly?"

st.markdown("---")


# -----------------------------------------------------------------------------
# 7. Conversational Chat Interface
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    avatar_val = "👨‍💻" if is_user else (IMAGE_PATH if IMAGE_PATH.exists() else "⚡")

    with st.chat_message(msg["role"], avatar=avatar_val):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Verified Resume Citations", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-title">Citation [{idx}] — {src.get('source', 'Karan Bhardwaj Resume')}</div>
                            <div class="citation-text">{src.get('content')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Input handler
typed_query = st.chat_input("Ask a question about Karan Bhardwaj...")
active_query = st.session_state.prompt_query or typed_query

if st.session_state.prompt_query:
    st.session_state.prompt_query = None

if active_query:
    if not groq_api_key:
        st.error("Please provide a valid Groq API Key in the sidebar.")
    elif not st.session_state.rag_chain:
        st.warning("Knowledge base is initializing. Please check the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(active_query)

        # Stream assistant response
        assistant_avatar = IMAGE_PATH if IMAGE_PATH.exists() else "⚡"
        with st.chat_message("assistant", avatar=assistant_avatar):
            box = st.empty()
            full_text = ""
            retrieved_docs = []

            try:
                for chunk in st.session_state.rag_chain.stream_query(
                    active_query, session_id=st.session_state.session_id
                ):
                    if chunk["type"] == "token":
                        full_text += chunk["content"]
                        box.markdown(full_text + " ▌")
                    elif chunk["type"] == "sources":
                        retrieved_docs = chunk["documents"]

                box.markdown(full_text)

                # Render sources
                citation_list = []
                if retrieved_docs:
                    with st.expander("Verified Resume Citations", expanded=False):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            src_name = doc.metadata.get("source", "Karan Bhardwaj Resume")
                            text_content = doc.page_content.strip()
                            citation_list.append({"source": src_name, "content": text_content})
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-title">Citation [{idx}] — {src_name}</div>
                                    <div class="citation-text">{text_content}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_text,
                        "sources": citation_list,
                    }
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# 8. Minimalist Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="minimal-footer">
        <strong>Karan Bhardwaj</strong> • Full Stack &amp; AI Systems Engineer<br>
        <a href="https://karanbhardwaj.in" target="_blank">Portfolio</a> ·
        <a href="https://linkedin.com/in/karan-bhardwaj" target="_blank">LinkedIn</a> ·
        <a href="https://github.com/karanongit" target="_blank">GitHub</a> ·
        <a href="mailto:karanbhardwaj1107@gmail.com">karanbhardwaj1107@gmail.com</a> ·
        <a href="tel:+916202640773">+91 6202640773</a><br>
        <span style="font-size:0.72rem; color:#94A3B8; margin-top:0.35rem; display:inline-block;">
            Built with LangChain LCEL &amp; Groq LPU™
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
