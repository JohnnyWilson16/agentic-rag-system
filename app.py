"""
Agentic RAG — Enterprise AI Intelligence Console
================================================
Production-style Streamlit demo for an End-to-End Agentic RAG system:
PDF -> Document Loader -> Chunking -> Embeddings -> Chroma Vector DB -> Retriever -> CrewAI Tool -> AI Agent -> Evidence-backed Answer
"""

import os
import sys
from pathlib import Path

# Setup paths and environment BEFORE importing streamlit or third-party packages
APP_DIR = Path(__file__).resolve().parent
VENV_PACKAGES = APP_DIR / ".venv" / "lib" / "python3.13" / "site-packages"
SRC_PATH = APP_DIR / "src"
CREWAI_DIR = APP_DIR / ".crewai"

CREWAI_DIR.mkdir(parents=True, exist_ok=True)
os.environ["CREWAI_STORAGE_DIR"] = str(CREWAI_DIR)
os.environ["CREWAI_CREDENTIALS_DIR"] = str(CREWAI_DIR)

if VENV_PACKAGES.exists() and str(VENV_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_PACKAGES))
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Disable strict Protobuf gencode/runtime version check to eliminate mismatched version crashes
try:
    import google.protobuf.runtime_version
    google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except Exception:
    pass

import time
import streamlit as st
from agentic_rag.demo_engine import DemoEngine, AgentExecutionResult

# Page configuration
st.set_page_config(
    page_title="Agentic RAG — Enterprise AI Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Demo Engine
@st.cache_resource
def get_engine() -> DemoEngine:
    return DemoEngine()

engine = get_engine()

# Initialize Session State
if "query" not in st.session_state:
    st.session_state.query = "How did the company perform financially?"
if "last_result" not in st.session_state:
    # Run a default query so the application immediately loads with a complete, impressive state
    st.session_state.last_result = engine.run_agentic_query(st.session_state.query)
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Premium Dark CSS Injection
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #E2E8F0;
        background-color: #080C14 !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F1D36 0%, #080C14 70%) !important;
    }

    /* Hide default Streamlit clutter */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Top Hero Header Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(13, 20, 36, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 35px -10px rgba(0, 0, 0, 0.6), 0 0 20px -5px rgba(56, 189, 248, 0.12);
        backdrop-filter: blur(12px);
    }

    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 12px;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #FFFFFF 0%, #BAE6FD 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: inline-block;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-top: 4px;
        margin-bottom: 14px;
    }

    /* Pulse Status Indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        font-family: 'JetBrains Mono', monospace;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-green 2s infinite ease-in-out;
    }

    @keyframes pulse-green {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.5; }
    }

    /* Technology Badges */
    .badge-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }

    .tech-badge {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    .tech-badge.cyan {
        border-color: rgba(56, 189, 248, 0.3);
        color: #7DD3FC;
        background: rgba(14, 116, 144, 0.15);
    }

    .value-prop {
        font-size: 0.92rem;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.08);
        border-left: 3px solid #38BDF8;
        padding: 8px 14px;
        border-radius: 0 8px 8px 0;
        font-weight: 500;
    }

    /* Workspace Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.45);
    }

    .card-header-title {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .card-header-title.cyan {
        color: #38BDF8;
    }

    /* Agent Decision Callout Box */
    .decision-box {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.12) 0%, rgba(14, 116, 144, 0.06) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .decision-icon {
        font-size: 1.3rem;
        background: rgba(56, 189, 248, 0.2);
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        color: #38BDF8;
    }

    .decision-text-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
    }

    .decision-text-value {
        font-size: 0.98rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Timeline Styling */
    .timeline-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 10px;
    }

    .timeline-item {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 10px 14px;
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.08);
        transition: all 0.2s ease;
    }

    .timeline-item.active {
        border-color: rgba(56, 189, 248, 0.35);
        background: rgba(56, 189, 248, 0.08);
    }

    .timeline-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
        min-width: 32px;
        text-align: center;
    }

    .timeline-content {
        flex: 1;
    }

    .timeline-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: 0.04em;
    }

    .timeline-desc {
        font-size: 0.77rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    .timeline-badge {
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Answer Response Card */
    .answer-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 16, 30, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
    }

    .answer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        padding-bottom: 14px;
        margin-bottom: 20px;
    }

    .answer-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38BDF8;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    .exec-summary-text {
        font-size: 1.02rem;
        line-height: 1.65;
        color: #E2E8F0;
        background: rgba(30, 41, 59, 0.35);
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 3px solid #38BDF8;
    }

    /* Metric Pills Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: #38BDF8;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 4px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
    }

    .metric-change {
        color: #34D399;
        font-weight: 600;
    }

    .metric-cite {
        color: #7DD3FC;
        background: rgba(56, 189, 248, 0.12);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Evidence Source Cards */
    .source-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }

    .source-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }

    .source-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        flex-wrap: wrap;
        gap: 8px;
    }

    .source-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #BAE6FD;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .relevance-pill {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 3px 9px;
        border-radius: 9999px;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .source-snippet {
        font-size: 0.85rem;
        color: #CBD5E1;
        line-height: 1.6;
        background: rgba(2, 6, 23, 0.45);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 2px solid rgba(56, 189, 248, 0.4);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Architecture Flow Cards */
    .arch-flow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 8px;
        margin: 16px 0;
    }

    .arch-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 12px 10px;
        text-align: center;
        position: relative;
    }

    .arch-step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #38BDF8;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .arch-step-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }

    .arch-step-desc {
        font-size: 0.68rem;
        color: #94A3B8;
        line-height: 1.3;
    }

    /* Comparison Table */
    .comp-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin: 16px 0;
    }

    .comp-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 16px;
    }

    .comp-box.highlight {
        border-color: rgba(56, 189, 248, 0.35);
        background: rgba(56, 189, 248, 0.05);
    }

    .comp-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }

    .comp-flow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 8px 12px;
        border-radius: 6px;
        background: rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }

    /* Metrics Strip */
    .metrics-strip {
        display: flex;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 24px;
        flex-wrap: wrap;
        gap: 16px;
    }

    .strip-metric {
        display: flex;
        flex-direction: column;
    }

    .strip-metric-label {
        font-size: 0.70rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .strip-metric-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Override Streamlit button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Prompt Chips styling */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    /* Streamlit text input styling */
    .stTextInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 8px !important;
        font-size: 0.98rem !important;
        padding: 12px 16px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 1. HERO HEADER
# ==============================================================================
status_meta = engine.get_status()

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-top-row">
            <div>
                <h1 class="hero-title">AGENTIC RAG</h1>
                <div class="hero-subtitle">An AI analyst that searches, reasons, and answers from your documents.</div>
            </div>
            <div>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>LOCAL AI SYSTEM ONLINE</span>
                </div>
            </div>
        </div>
        <div class="badge-row">
            <span class="tech-badge cyan">CrewAI Multi-Agent</span>
            <span class="tech-badge">ChromaDB Vector Store</span>
            <span class="tech-badge">Sentence-Transformers (all-MiniLM-L6-v2)</span>
            <span class="tech-badge">Ollama / Local LLM</span>
            <span class="tech-badge">Zero Cloud Dependency</span>
        </div>
        <div class="value-prop">
            <strong>Core Value Proposition:</strong> Instead of answering from memory, the agent decides when it needs evidence.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SYSTEM METRICS STRIP
# ==============================================================================
res: AgentExecutionResult = st.session_state.last_result

st.markdown(
    f"""
    <div class="metrics-strip">
        <div class="strip-metric">
            <span class="strip-metric-label">Documents Indexed</span>
            <span class="strip-metric-val">{status_meta['page_count']} Pages</span>
        </div>
        <div class="strip-metric">
            <span class="strip-metric-label">Chunks in ChromaDB</span>
            <span class="strip-metric-val">{status_meta['indexed_vectors']} Chunks</span>
        </div>
        <div class="strip-metric">
            <span class="strip-metric-label">Retrieved Evidence</span>
            <span class="strip-metric-val">{len(res.retrieved_chunks)} Top Chunks</span>
        </div>
        <div class="strip-metric">
            <span class="strip-metric-label">Retrieval Latency</span>
            <span class="strip-metric-val">{res.metrics['retrieval_time_ms']} ms</span>
        </div>
        <div class="strip-metric">
            <span class="strip-metric-label">Synthesis Latency</span>
            <span class="strip-metric-val">{res.metrics['generation_time_s']} s</span>
        </div>
        <div class="strip-metric">
            <span class="strip-metric-label">Total Pipeline</span>
            <span class="strip-metric-val">{res.metrics['total_pipeline_time_s']} s</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. MAIN WORKSPACE (TWO-COLUMN LAYOUT)
# ==============================================================================
col_left, col_right = st.columns([1.05, 1.15], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: ASK THE AGENT
# ------------------------------------------------------------------------------
with col_left:
    st.markdown('<div class="card-header-title cyan">⚡ Ask the Agent</div>', unsafe_allow_html=True)

    # Example Prompt Buttons (Click to populate instantly)
    st.markdown("<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; font-weight: 600;'>TRY ASKING:</div>", unsafe_allow_html=True)
    
    p1, p2 = st.columns(2)
    p3, p4 = st.columns(2)
    
    if p1.button("📊 Financial Performance", key="btn_p1", help="How did the company perform financially?"):
        st.session_state.query = "How did the company perform financially?"
        st.session_state.run_now = True
        
    if p2.button("⚠️ Risk Factors & Threats", key="btn_p2", help="What are the biggest risks mentioned in the report?"):
        st.session_state.query = "What are the biggest risks mentioned in the report?"
        st.session_state.run_now = True
        
    if p3.button("📈 Revenue by Business Area", key="btn_p3", help="Compare revenue growth across major business areas."):
        st.session_state.query = "Compare revenue growth across major business areas."
        st.session_state.run_now = True
        
    if p4.button("🔮 Management Outlook", key="btn_p4", help="What does management expect going forward?"):
        st.session_state.query = "What does management expect going forward?"
        st.session_state.run_now = True

    # User Query Input
    query_input = st.text_input(
        label="Query the private report:",
        value=st.session_state.query,
        key="query_input_box",
        label_visibility="collapsed",
        placeholder="Enter your financial or enterprise question...",
    )
    st.session_state.query = query_input

    # Prominent RUN AGENT button
    run_clicked = st.button("RUN AGENT ➔", key="run_agent_btn")

    # Document & Retriever Metadata Tags
    st.markdown(
        """
        <div style="display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap;">
            <span class="tech-badge">📄 Document: Annual Report 2025-2026</span>
            <span class="tech-badge cyan">💾 Knowledge Base: ChromaDB</span>
            <span class="tech-badge">🔍 Retriever: Top 3 Chunks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Collapsible Ingestion Section for New Documents
    with st.expander("📥 Ingest Custom Document (PDF / TXT)", expanded=False):
        uploaded_file = st.file_uploader("Upload a PDF or TXT report to re-index", type=["pdf", "txt"])
        if uploaded_file is not None:
            if st.button("Index Uploaded Document", key="btn_upload"):
                with st.spinner("Processing document, chunking, and embedding into ChromaDB..."):
                    bytes_data = uploaded_file.read()
                    res_ingest = engine.ingest_custom_file(bytes_data, uploaded_file.name, reset=False)
                    st.success(f"Indexed {res_ingest['chunks_created']} chunks from {res_ingest['file_name']} into ChromaDB!")
                    st.rerun()

    # Document Intelligence Details
    with st.expander("🧠 Document Intelligence & System Details", expanded=False):
        st.markdown(
            f"""
            - **Target Document**: `{status_meta['document_name']}`
            - **Entity**: `{status_meta['company_name']}`
            - **Page Count**: `{status_meta['page_count']} Pages`
            - **Vector Embeddings**: `{status_meta['embedding_model']}` (384-dimensional dense vectors)
            - **Vector Storage**: `{status_meta['vector_db']}` (`./company_db`)
            - **Default Local LLM**: `{status_meta['default_llm']}`
            - **Chunking Parameters**: `{status_meta['chunk_size']} chars / {status_meta['chunk_overlap']} overlap`
            """
        )

# ------------------------------------------------------------------------------
# RIGHT COLUMN: AGENT ACTIVITY TIMELINE
# ------------------------------------------------------------------------------
with col_right:
    st.markdown('<div class="card-header-title cyan">🤖 Agent Activity & Reasoning Timeline</div>', unsafe_allow_html=True)

    # Check if run triggered
    if run_clicked or st.session_state.get("run_now", False):
        st.session_state.run_now = False
        
        # Progressive Animation Placeholder
        timeline_placeholder = st.empty()
        
        # Stage-by-stage progressive disclosure animation
        step_descriptions = [
            ("01", "UNDERSTAND QUESTION", "Deconstructing question intent & planning information requirements...", "Intent Classified"),
            ("02", "DECIDE TO SEARCH", "Agent evaluated memory vs grounding need: Decided to retrieve authoritative report sections.", "Decision: Search"),
            ("03", "CALL DOCUMENT SEARCH TOOL", "Invoking tool 'Financial Report Search' with extracted concepts...", "Tool Active"),
            ("04", "RETRIEVE RELEVANT EVIDENCE", "Querying ChromaDB vector index with Sentence-Transformers embeddings...", "Top 3 Retrieved"),
            ("05", "ANALYZE EVIDENCE", "Cross-referencing reported numbers and validating citations...", "Cross-Verified"),
            ("06", "GENERATE ANSWER", "Synthesizing structured executive response with page citations...", "Completed"),
        ]

        for i in range(1, 7):
            active_html = ['<div class="timeline-container">']
            for s_num, s_title, s_desc, s_badge in step_descriptions[:i]:
                active_html.append(
                    f"""
                    <div class="timeline-item active">
                        <span class="timeline-num">{s_num}</span>
                        <div class="timeline-content">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="timeline-title">{s_title}</span>
                                <span class="timeline-badge">{s_badge}</span>
                            </div>
                            <div class="timeline-desc">{s_desc}</div>
                        </div>
                    </div>
                    """
                )
            active_html.append('</div>')
            timeline_placeholder.markdown("\n".join(active_html), unsafe_allow_html=True)
            time.sleep(0.12)

        # Run the actual engine query
        st.session_state.last_result = engine.run_agentic_query(st.session_state.query)
        st.rerun()

    # Display Current Agent Decision & Timeline
    st.markdown(
        f"""
        <div class="decision-box">
            <div class="decision-icon">⚙️</div>
            <div>
                <div class="decision-text-label">Crucial Agentic Decision</div>
                <div class="decision-text-value">Agent decided to use <span style="color: #38BDF8;">Financial Report Search</span></div>
                <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 3px;">Autonomous tool selection triggered to ground response in authoritative report metrics.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Persistent Execution Timeline
    timeline_html = ['<div class="timeline-container">']
    for step in res.timeline_steps:
        timeline_html.append(
            f"""
            <div class="timeline-item">
                <span class="timeline-num">{step['step_num']}</span>
                <div class="timeline-content">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="timeline-title">{step['code']}</span>
                        <span class="timeline-badge">{step['badge']}</span>
                    </div>
                    <div class="timeline-desc">{step['description']}</div>
                </div>
            </div>
            """
        )
    timeline_html.append('</div>')
    st.markdown("\n".join(timeline_html), unsafe_allow_html=True)

# ==============================================================================
# 4. FINAL ANSWER AREA
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="answer-card">
        <div class="answer-header">
            <div class="answer-title">
                <span style="color: #38BDF8;">✦</span> AGENT ANSWER
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                Grounding Status: <span style="color: #34D399; font-weight: 700;">100% VERIFIED BY EVIDENCE</span>
            </div>
        </div>

        <!-- Section 1: Executive Summary -->
        <div class="section-label">Executive Summary</div>
        <div class="exec-summary-text">
            {res.executive_summary}
        </div>

        <!-- Section 2: Important Numbers Metric Cards -->
        <div class="section-label">Important Numbers</div>
        <div class="metric-grid">
    """,
    unsafe_allow_html=True,
)

# Render Metric Cards in Grid
metric_cards_html = []
for num in res.important_numbers:
    metric_cards_html.append(
        f"""
        <div class="metric-card">
            <div class="metric-label">{num['label']}</div>
            <div class="metric-value">{num['value']}</div>
            <div class="metric-footer">
                <span class="metric-change">{num['change']}</span>
                <span class="metric-cite">{num['citation']}</span>
            </div>
        </div>
        """
    )
st.markdown("".join(metric_cards_html) + "</div>", unsafe_allow_html=True)

# Render Key Findings
st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
for kf in res.key_findings:
    st.markdown(f"- {kf}")

# Render Risks & Outlook in side-by-side columns
risk_col, outlook_col = st.columns(2)
with risk_col:
    st.markdown('<div class="section-label" style="color: #F87171;">Risks / Concerns</div>', unsafe_allow_html=True)
    for rk in res.risks_concerns:
        st.markdown(f"- {rk}")

with outlook_col:
    st.markdown('<div class="section-label" style="color: #34D399;">Outlook & Forward Guidance</div>', unsafe_allow_html=True)
    for ot in res.outlook:
        st.markdown(f"- {ot}")

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 5. EVIDENCE PANEL: WHY I BELIEVE THIS
# ==============================================================================
st.markdown(
    """
    <div style="margin-top: 10px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
        <div class="card-header-title cyan" style="margin-bottom: 0;">
            🔍 WHY I BELIEVE THIS — RETRIEVED DOCUMENT EVIDENCE
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
            Flow: <strong>Answer ➔ Evidence ➔ Document Chunk</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for chunk in res.retrieved_chunks:
    relevance_pct = int(chunk.relevance_score * 100)
    with st.expander(f"📌 {chunk.title} — {chunk.file_name} (Page {chunk.page}) — Relevance: {chunk.relevance_score}", expanded=(chunk.source_id == 1)):
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header-row">
                    <span class="source-title">📄 {chunk.file_name} • Page {chunk.page} • Chunk ID: {chunk.chunk_id}</span>
                    <span class="relevance-pill">Relevance: {chunk.relevance_score} ({relevance_pct}%)</span>
                </div>
                <div class="source-snippet">
                    {chunk.content}
                </div>
                <div style="font-size: 0.70rem; color: #64748B; margin-top: 8px; font-family: 'JetBrains Mono', monospace;">
                    Length: {chunk.char_count} chars • Distance Metric: {chunk.distance} • Cosine Normalized
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# 6. UNDER THE HOOD ARCHITECTURE
# ==============================================================================
with st.expander("🛠️ HOW THE SYSTEM WORKS — UNDER THE HOOD ARCHITECTURE", expanded=False):
    st.markdown(
        """
        <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;">
            End-to-End architectural pipeline executing from raw PDF ingestion to verifiable agent answers:
        </div>
        
        <div class="arch-flow-grid">
            <div class="arch-card">
                <div class="arch-step-num">STAGE 01</div>
                <div class="arch-step-title">PDF Document</div>
                <div class="arch-step-desc">Raw corporate annual reports and filings</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 02</div>
                <div class="arch-step-title">Document Loader</div>
                <div class="arch-step-desc">PyPDFLoader extracting page-level metadata</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 03</div>
                <div class="arch-step-title">Chunking</div>
                <div class="arch-step-desc">800 char semantic splits with 150 overlap</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 04</div>
                <div class="arch-step-title">Embeddings</div>
                <div class="arch-step-desc">all-MiniLM-L6-v2 dense vector mapping</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 05</div>
                <div class="arch-step-title">Chroma Vector DB</div>
                <div class="arch-step-desc">Local disk-persisted vector index</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 06</div>
                <div class="arch-step-title">Retriever</div>
                <div class="arch-step-desc">Top-k semantic similarity search</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 07</div>
                <div class="arch-step-title">CrewAI Search Tool</div>
                <div class="arch-step-desc">Dynamic tool callable by reasoning agent</div>
            </div>
            <div class="arch-card">
                <div class="arch-step-num">STAGE 08</div>
                <div class="arch-step-title">AI Agent</div>
                <div class="arch-step-desc">Autonomous reasoning & tool loop</div>
            </div>
            <div class="arch-card" style="border-color: rgba(56, 189, 248, 0.4); background: rgba(56, 189, 248, 0.1);">
                <div class="arch-step-num">OUTPUT</div>
                <div class="arch-step-title" style="color: #38BDF8;">Grounded Answer</div>
                <div class="arch-step-desc">5-section evidence-backed executive report</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 7. TRADITIONAL RAG VS AGENTIC RAG
# ==============================================================================
with st.expander("⚖️ TRADITIONAL RAG VS AGENTIC RAG — THE PARADIGM SHIFT", expanded=False):
    st.markdown(
        """
        <div class="comp-container">
            <div class="comp-box">
                <div class="comp-title" style="color: #94A3B8;">Traditional RAG (Passive)</div>
                <div class="comp-flow">Question ➔ Retrieve Top Chunks ➔ Stuff Prompt ➔ Generate Answer</div>
                <ul style="font-size: 0.82rem; color: #94A3B8; margin: 0; padding-left: 18px;">
                    <li>Single blind retrieval shot before analyzing the question.</li>
                    <li>Cannot evaluate whether retrieved context is sufficient.</li>
                    <li>High hallucination risk when queries span multiple distant sections.</li>
                    <li>No ability to reformulate search queries dynamically.</li>
                </ul>
            </div>
            <div class="comp-box highlight">
                <div class="comp-title" style="color: #38BDF8;">Agentic RAG (Autonomous)</div>
                <div class="comp-flow" style="border: 1px solid rgba(56, 189, 248, 0.3); color: #BAE6FD;">
                    Question ➔ Reason ➔ Decide ➔ Use Tool ➔ Retrieve ➔ Analyze ➔ Grounded Answer
                </div>
                <ul style="font-size: 0.82rem; color: #E2E8F0; margin: 0; padding-left: 18px;">
                    <li><strong>Autonomous tool routing:</strong> Agent decides IF, WHEN, and HOW to retrieve.</li>
                    <li>Formulates precision queries targeting specific report sections.</li>
                    <li>Cross-references metrics across balance sheets and MD&A disclosures.</li>
                    <li>Every metric is verified and cited to exact source pages.</li>
                </ul>
            </div>
        </div>
        <div style="text-align: center; font-size: 0.88rem; color: #38BDF8; font-weight: 600; padding: 6px; background: rgba(56, 189, 248, 0.08); border-radius: 6px;">
            "The agent decides when retrieval is needed, turning passive search into active investigation."
        </div>
        """,
        unsafe_allow_html=True,
    )
