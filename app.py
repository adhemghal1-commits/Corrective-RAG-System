"""
Corrective RAG System — Streamlit Application
===============================================
A full-featured UI for document-based question answering
with corrective retrieval and answer verification.
"""

import streamlit as st
from pathlib import Path
import time

from config import (
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
    RELEVANCE_THRESHOLD,
    TOP_K,
    MAX_REWRITE_ATTEMPTS,
)
from document_processor import (
    process_all_documents,
    process_uploaded_file,
    process_document,
)
from embedding_store import EmbeddingStore
from corrective_rag_pipeline import CorrectiveRAGPipeline
from retrieval_evaluator import ContextQuality


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Corrective RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .source-badge {
        display: inline-block;
        background: #e8f4f8;
        color: #1a73e8;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        border: 1px solid #bee5eb;
    }
    .eval-step {
        border-left: 3px solid #667eea;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    .quality-strong { color: #28a745; font-weight: 700; }
    .quality-weak { color: #ffc107; font-weight: 700; }
    .quality-irrelevant { color: #dc3545; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
def init_session_state():
    if "store" not in st.session_state:
        st.session_state.store = EmbeddingStore()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = CorrectiveRAGPipeline(st.session_state.store)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

init_session_state()


# ──────────────────────────────────────────────
# Sidebar — Document Management
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Document Management")
    st.markdown("---")

    # ── Load pre-existing documents ──
    st.markdown("### 📂 Course Documents")

    if DOCUMENTS_DIR.exists():
        doc_files = [
            f for f in DOCUMENTS_DIR.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if doc_files:
            st.info(f"Found **{len(doc_files)}** documents in `Course files/`")
            for f in doc_files:
                st.markdown(f"- 📄 {f.name}")

            if st.button("🔄 Index Course Documents", use_container_width=True):
                with st.spinner("Processing and indexing documents..."):
                    chunks = process_all_documents(str(DOCUMENTS_DIR))
                    n = st.session_state.pipeline.ingest_documents(chunks)
                    st.session_state.documents_loaded = True
                st.success(f"✅ Indexed **{n}** chunks from {len(doc_files)} documents!")
                st.rerun()
        else:
            st.warning("No supported documents found in `Course files/`.")
    else:
        st.warning("`Course files/` directory not found.")

    st.markdown("---")

    # ── Upload new documents ──
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or MD files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Index Uploaded Files", use_container_width=True):
        with st.spinner("Processing uploaded files..."):
            total_chunks = 0
            for uf in uploaded_files:
                chunks = process_uploaded_file(uf)
                n = st.session_state.pipeline.ingest_documents(chunks)
                total_chunks += n
                st.markdown(f"  ✅ {uf.name}: {n} chunks")
            st.session_state.documents_loaded = True
        st.success(f"✅ Indexed **{total_chunks}** chunks from {len(uploaded_files)} files!")
        st.rerun()

    st.markdown("---")

    # ── Vector Store Stats ──
    st.markdown("### 📊 Vector Store Status")
    stats = st.session_state.pipeline.get_stats()
    is_ready = st.session_state.pipeline.is_ready()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chunks", stats["total_chunks"])
    with col2:
        status = "🟢 Ready" if is_ready else "🔴 Empty"
        st.metric("Status", status)

    st.caption(f"Model: `{stats['embedding_model']}`")

    if is_ready and st.button("🗑️ Clear Vector Store", use_container_width=True):
        st.session_state.pipeline.clear()
        st.session_state.documents_loaded = False
        st.success("Vector store cleared.")
        st.rerun()

    st.markdown("---")

    # ── Settings ──
    st.markdown("### ⚙️ Settings")
    st.caption(f"Top-K retrieval: **{TOP_K}**")
    st.caption(f"Relevance threshold: **{RELEVANCE_THRESHOLD}**")
    st.caption(f"Max rewrite attempts: **{MAX_REWRITE_ATTEMPTS}**")


# ──────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────

# Header
st.markdown('<div class="main-header">🔍 Corrective RAG System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Ask questions about your documents. The system retrieves relevant context, '
    'evaluates its quality, rewrites the query if needed, and generates verified answers.'
    '</div>',
    unsafe_allow_html=True,
)

# Check if ready
if not st.session_state.pipeline.is_ready():
    st.warning(
        "⚠️ **No documents indexed yet.** "
        "Please index the course documents or upload files using the sidebar."
    )

# ── Chat History Display ──
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

        # Expandable details
        with st.expander("📋 Details", expanded=False):
            # Sources
            if entry.get("sources"):
                st.markdown("**📎 Sources:**")
                for src in entry["sources"]:
                    st.markdown(f'<span class="source-badge">📄 {src}</span>', unsafe_allow_html=True)

            # Evaluation trace
            if entry.get("evaluation_history"):
                st.markdown("**🔍 Retrieval Evaluation Trace:**")
                for ev in entry["evaluation_history"]:
                    quality_class = {
                        "strong": "quality-strong",
                        "weak": "quality-weak",
                        "irrelevant": "quality-irrelevant",
                    }.get(ev["quality"], "")
                    st.markdown(
                        f'<div class="eval-step">'
                        f'<strong>Attempt {ev["attempt"]}</strong> — '
                        f'Query: <em>"{ev["query"][:80]}..."</em><br>'
                        f'Quality: <span class="{quality_class}">{ev["quality"].upper()}</span> | '
                        f'Relevant: {ev["num_relevant"]}/{ev["total_retrieved"]} | '
                        f'Avg Similarity: {ev["avg_similarity"]:.3f}<br>'
                        f'<small>{ev["explanation"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Rewrite info
            if entry.get("was_rewritten"):
                st.info(
                    f"🔄 Query was rewritten **{entry['rewrite_attempts']}** time(s).\n\n"
                    f"Final query: *\"{entry['final_query']}\"*"
                )


# ── Chat Input ──
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.pipeline.is_ready(),
)

if question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Run the pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and evaluating context..."):
            start_time = time.time()
            result = st.session_state.pipeline.run(question)
            elapsed = time.time() - start_time

        # Display the answer
        st.markdown(result.answer)

        # Display metadata
        with st.expander("📋 Details", expanded=True):
            # Timing
            st.caption(f"⏱️ Completed in {elapsed:.1f}s")

            # Sources
            if result.sources:
                st.markdown("**📎 Sources:**")
                for src in result.sources:
                    st.markdown(
                        f'<span class="source-badge">📄 {src}</span>',
                        unsafe_allow_html=True,
                    )

            # Evaluation trace
            st.markdown("**🔍 Retrieval Evaluation Trace:**")
            for ev in result.evaluation_history:
                quality_class = {
                    "strong": "quality-strong",
                    "weak": "quality-weak",
                    "irrelevant": "quality-irrelevant",
                }.get(ev["quality"], "")
                st.markdown(
                    f'<div class="eval-step">'
                    f'<strong>Attempt {ev["attempt"]}</strong> — '
                    f'Query: <em>"{ev["query"][:80]}..."</em><br>'
                    f'Quality: <span class="{quality_class}">{ev["quality"].upper()}</span> | '
                    f'Relevant: {ev["num_relevant"]}/{ev["total_retrieved"]} | '
                    f'Avg Similarity: {ev["avg_similarity"]:.3f}<br>'
                    f'<small>{ev["explanation"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Rewrite info
            if result.was_rewritten:
                st.info(
                    f"🔄 Query was rewritten **{result.rewrite_attempts}** time(s).\n\n"
                    f"Final query: *\"{result.final_query}\"*"
                )

            # Answerable status
            if not result.answerable:
                st.warning(
                    "⚠️ The system could not find a confident answer in the documents."
                )

    # Save to chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer": result.answer,
        "sources": result.sources,
        "answerable": result.answerable,
        "was_rewritten": result.was_rewritten,
        "rewrite_attempts": result.rewrite_attempts,
        "final_query": result.final_query,
        "evaluation_history": result.evaluation_history,
    })


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #888; font-size: 0.85rem;">'
    'Corrective RAG System — Retrieval-Augmented Generation with Quality Evaluation & Query Correction'
    '</div>',
    unsafe_allow_html=True,
)

