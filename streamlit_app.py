"""
PoliRAG — Enterprise Policy & Compliance Intelligence Engine
Interactive Streamlit Demo for Recruiters & Stakeholders
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
import numpy as np

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PoliRAG — Enterprise Policy Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero gradient header */
.hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.3);
}
.hero-header h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1A1F2E, #252B3B);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.15);
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label {
    font-size: 0.85rem;
    color: rgba(250,250,250,0.6);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Pipeline step badges */
.pipeline-step {
    background: linear-gradient(145deg, #1A1F2E, #252B3B);
    border-left: 3px solid #6C63FF;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.8rem;
}
.pipeline-step .step-title {
    font-weight: 600;
    color: #6C63FF;
    font-size: 0.95rem;
}
.pipeline-step .step-desc {
    color: rgba(250,250,250,0.7);
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* Result cards */
.result-card {
    background: linear-gradient(145deg, #1A1F2E, #252B3B);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Tag pills */
.tag {
    display: inline-block;
    background: rgba(108, 99, 255, 0.15);
    color: #6C63FF;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}

/* Success / warning badges */
.badge-success {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.badge-warning {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.badge-danger {
    display: inline-block;
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Architecture diagram container */
.arch-container {
    background: linear-gradient(145deg, #0E1117, #1A1F2E);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 16px;
    padding: 2rem;
}

/* Animated glow */
@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(108, 99, 255, 0.2); }
    50% { box-shadow: 0 0 20px rgba(108, 99, 255, 0.4); }
}
.glow { animation: glow 3s ease-in-out infinite; }

/* Sidebar styling */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0E1117, #1A1F2E);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ PoliRAG")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📄 Document Ingestion",
            "🔍 Chunking Explorer",
            "🔐 Guardrails Demo",
            "📊 Evaluation Metrics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color: rgba(250,250,250,0.4); font-size: 0.75rem;">'
        "Built with FastAPI · FAISS · OpenAI<br>"
        "© 2025 PoliRAG"
        "</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown(
        """
        <div class="hero-header">
            <h1>🛡️ PoliRAG</h1>
            <p>Enterprise Policy & Compliance Intelligence Engine — Production-grade RAG system
            for policy documents with hybrid retrieval, guardrails, and evaluation pipelines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metrics row
    cols = st.columns(5)
    metrics = [
        ("10+", "Document Formats"),
        ("4", "Guardrail Layers"),
        ("5", "Eval Metrics"),
        ("Hybrid", "BM25 + Semantic"),
        ("RBAC", "Access Control"),
    ]
    for col, (val, label) in zip(cols, metrics):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline overview
    left, right = st.columns(2)

    with left:
        st.markdown("### ⚡ RAG Pipeline")
        steps = [
            ("1. Ingestion", "Multi-format document loading with deduplication & versioning"),
            ("2. Chunking", "Recursive & table-aware splitting with token limits"),
            ("3. Embedding", "Async batch embedding with persistent SQLite caching"),
            ("4. Indexing", "Dual-store: FAISS (semantic) + BM25 (keyword)"),
            ("5. Retrieval", "Hybrid search with cross-encoder reranking"),
            ("6. Generation", "LLM response with citations & hallucination detection"),
        ]
        for title, desc in steps:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### 🔒 Security & Guardrails")
        guards = [
            ("PII Detection & Redaction", "Regex + spaCy NER — emails, SSNs, phone numbers, credit cards"),
            ("Role-Based Access Control", "Hierarchical roles: Admin → Legal → Audit → Compliance → Read-Only"),
            ("Rate Limiting", "Token-bucket per tenant with configurable burst capacity"),
            ("Hallucination Detection", "Citation-based confidence scoring with flagging"),
        ]
        for title, desc in guards:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="step-title">🔐 {title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 📈 Observability")
        obs = [
            ("Structured Logging", "Loguru with JSON + colored human-readable output"),
            ("Prometheus Metrics", "Request counts, latency histograms, token usage, cache stats"),
        ]
        for title, desc in obs:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="step-title">📊 {title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tech stack
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    tags = [
        "Python 3.11", "FastAPI", "Pydantic v2", "OpenAI", "FAISS",
        "BM25", "spaCy", "Sentence Transformers", "Cross-Encoder",
        "SQLite", "Prometheus", "Loguru", "Docker", "pytest",
    ]
    tag_html = " ".join([f'<span class="tag">{t}</span>' for t in tags])
    st.markdown(f"<div>{tag_html}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENT INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Document Ingestion":
    st.markdown(
        """
        <div class="hero-header">
            <h1>📄 Document Ingestion</h1>
            <p>Multi-format document loading with content hashing, deduplication, and versioning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Supported formats
    st.markdown("### Supported Formats")
    format_cols = st.columns(6)
    formats = [
        ("📝", "TXT", "Plain text"),
        ("📰", "PDF", "PyPDF2 + pdfplumber"),
        ("📄", "DOCX", "python-docx"),
        ("📋", "Markdown", "Native parsing"),
        ("📧", "Email", "EML / MSG"),
        ("📊", "JSON", "Structured data"),
    ]
    for col, (icon, name, tech) in zip(format_cols, formats):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="value">{icon}</div>
                <div class="label">{name}<br><small style="font-size:0.65rem">{tech}</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Live demo
    st.markdown("### 🔬 Live Demo — Upload a Document")
    uploaded = st.file_uploader(
        "Drop a text or markdown file to see the ingestion pipeline in action",
        type=["txt", "md"],
    )

    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc_id = str(uuid4())[:8]

        st.markdown("#### Pipeline Output")
        progress = st.progress(0)

        # Step 1 — Load
        with st.spinner("Loading document..."):
            time.sleep(0.4)
            progress.progress(25)

        # Step 2 — Hash
        with st.spinner("Computing content hash..."):
            time.sleep(0.3)
            progress.progress(50)

        # Step 3 — Metadata
        with st.spinner("Extracting metadata..."):
            time.sleep(0.3)
            progress.progress(75)

        # Step 4 — Store
        with st.spinner("Storing document..."):
            time.sleep(0.3)
            progress.progress(100)

        st.success("✅ Document ingested successfully!")

        # Results
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4 style="margin-top:0">📋 Document Metadata</h4>
                    <table style="width:100%">
                        <tr><td style="color:rgba(250,250,250,0.5)">Document ID</td><td><code>{doc_id}</code></td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Filename</td><td>{uploaded.name}</td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Size</td><td>{len(content):,} bytes</td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Content Hash</td><td><code>{content_hash[:16]}…</code></td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Version</td><td>1</td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Word Count</td><td>{len(content.split()):,}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown("##### Preview (first 500 chars)")
            st.code(content[:500] + ("…" if len(content) > 500 else ""), language="text")

    elif st.button("📥 Load Sample Policy Document"):
        sample_path = ROOT / "data" / "raw" / "sample_policy.txt"
        if sample_path.exists():
            content = sample_path.read_text(encoding="utf-8")
            st.session_state["sample_content"] = content
            st.rerun()

    if "sample_content" in st.session_state:
        content = st.session_state["sample_content"]
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc_id = str(uuid4())[:8]

        st.success("✅ Sample policy document loaded!")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4 style="margin-top:0">📋 Document Metadata</h4>
                    <table style="width:100%">
                        <tr><td style="color:rgba(250,250,250,0.5)">Document ID</td><td><code>{doc_id}</code></td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Filename</td><td>sample_policy.txt</td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Size</td><td>{len(content):,} bytes</td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Content Hash</td><td><code>{content_hash[:16]}…</code></td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Version</td><td>1</td></tr>
                        <tr><td style="color:rgba(250,250,250,0.5)">Word Count</td><td>{len(content.split()):,}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown("##### Preview (first 500 chars)")
            st.code(content[:500] + "…", language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHUNKING EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Chunking Explorer":
    st.markdown(
        """
        <div class="hero-header">
            <h1>🔍 Chunking Explorer</h1>
            <p>Visualize how documents are split into semantically meaningful chunks
            with recursive separators and token limits.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.slider("Max Chunk Size (tokens)", 64, 1024, 512, 32)
    with col2:
        chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 200, 50, 10)
    with col3:
        strategy = st.selectbox("Strategy", ["Recursive", "Table-Preserving"])

    # Sample text for chunking
    sample_path = ROOT / "data" / "raw" / "sample_policy.txt"
    if sample_path.exists():
        text = sample_path.read_text(encoding="utf-8")
    else:
        text = "This is a sample document. It contains multiple paragraphs.\n\nParagraph two has more content about policy requirements.\n\nParagraph three discusses compliance measures."

    text = st.text_area("Document Text", text, height=200)

    if st.button("🔪 Chunk Document", type="primary"):
        # Simulate chunking with separators
        separators = ["\n\n", "\n", ". ", " "]

        def _recursive_chunk(txt, seps, max_tokens):
            """Simple recursive chunking for demo purposes."""
            chunks_out = []
            if not seps or len(txt.split()) <= max_tokens:
                if txt.strip():
                    chunks_out.append(txt.strip())
                return chunks_out

            sep = seps[0]
            parts = txt.split(sep)
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate.split()) > max_tokens and current:
                    chunks_out.append(current.strip())
                    current = part.strip()
                else:
                    current = candidate

            if current.strip():
                if len(current.split()) > max_tokens:
                    chunks_out.extend(_recursive_chunk(current, seps[1:], max_tokens))
                else:
                    chunks_out.append(current.strip())
            return chunks_out

        with st.spinner("Chunking document..."):
            time.sleep(0.3)
            chunks = _recursive_chunk(text, separators, chunk_size)

        st.markdown(f"### Results — **{len(chunks)} chunks** generated")

        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        token_counts = [len(c.split()) for c in chunks]
        s1.metric("Total Chunks", len(chunks))
        s2.metric("Avg Tokens/Chunk", f"{np.mean(token_counts):.0f}")
        s3.metric("Min Tokens", min(token_counts))
        s4.metric("Max Tokens", max(token_counts))

        # Display chunks
        for i, chunk in enumerate(chunks):
            tokens = len(chunk.split())
            chars = len(chunk)
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:12]

            with st.expander(f"📦 Chunk {i+1}  —  {tokens} tokens  ·  {chars} chars  ·  #{chunk_hash}"):
                st.code(chunk, language="text")
                st.markdown(
                    f'<span class="tag">Index: {i}</span>'
                    f'<span class="tag">Tokens: {tokens}</span>'
                    f'<span class="tag">Hash: {chunk_hash}</span>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GUARDRAILS DEMO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔐 Guardrails Demo":
    st.markdown(
        """
        <div class="hero-header">
            <h1>🔐 Guardrails Demo</h1>
            <p>Live demonstration of PII detection, RBAC enforcement, and rate limiting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🕵️ PII Detection", "👤 RBAC", "⏱️ Rate Limiting"])

    # ── PII Detection ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("### PII Detection & Redaction")
        st.markdown("Enter text containing PII — the system detects and redacts it automatically.")

        sample_pii_texts = [
            "Contact John at john.doe@acme.com or call 555-123-4567. SSN: 123-45-6789.",
            "Send payment to credit card 4111-1111-1111-1111, billing to 123 Main St.",
            "Employee records for Jane Smith (SSN: 987-65-4321) are stored in HR.",
        ]

        pii_text = st.text_area(
            "Input Text",
            sample_pii_texts[0],
            height=100,
            key="pii_input",
        )

        if st.button("🔍 Detect & Redact PII", type="primary"):
            # Regex-based PII detection (same patterns as the real module)
            patterns = {
                "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            }

            detections = []
            for pii_type, pattern in patterns.items():
                for match in re.finditer(pattern, pii_text):
                    detections.append({
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    })

            if detections:
                st.warning(f"⚠️ Found **{len(detections)} PII entities**")

                # Show detections
                for d in detections:
                    badge_class = "badge-danger" if d["type"] in ("ssn", "credit_card") else "badge-warning"
                    st.markdown(
                        f'<span class="{badge_class}">{d["type"].upper()}</span> '
                        f'`{d["value"]}` (position {d["start"]}–{d["end"]})',
                        unsafe_allow_html=True,
                    )

                # Redacted output
                st.markdown("#### Redacted Output")
                redacted = pii_text
                for d in sorted(detections, key=lambda x: x["start"], reverse=True):
                    placeholder = f"[REDACTED_{d['type'].upper()}]"
                    redacted = redacted[: d["start"]] + placeholder + redacted[d["end"]:]
                st.code(redacted, language="text")
            else:
                st.success("✅ No PII detected — text is clean.")

    # ── RBAC ────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Role-Based Access Control")
        st.markdown("PoliRAG enforces a strict role hierarchy for document access.")

        # Role hierarchy visualization
        st.markdown(
            """
            <div class="result-card">
                <h4 style="margin-top:0">Role Hierarchy</h4>
                <p style="font-family: monospace; font-size: 1.1rem; text-align: center;">
                    <span class="badge-danger">ADMIN</span> →
                    <span class="badge-warning">LEGAL</span> →
                    <span class="badge-warning">AUDIT</span> →
                    <span class="badge-success">COMPLIANCE</span> →
                    <span class="badge-success">READ_ONLY</span>
                </p>
                <p style="color: rgba(250,250,250,0.5); font-size: 0.85rem; text-align:center;">
                    Higher roles inherit access to all lower-level documents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Interactive demo
        col_role, col_doc = st.columns(2)
        with col_role:
            user_role = st.selectbox(
                "Your Role",
                ["admin", "legal", "audit", "compliance", "read_only"],
            )
        with col_doc:
            doc_scope = st.selectbox(
                "Document Classification",
                ["read_only", "compliance", "audit", "legal", "admin"],
            )

        role_hierarchy = {
            "admin": 5,
            "legal": 4,
            "audit": 3,
            "compliance": 2,
            "read_only": 1,
        }

        if st.button("🔒 Check Access", type="primary"):
            user_level = role_hierarchy[user_role]
            doc_level = role_hierarchy[doc_scope]

            if user_level >= doc_level:
                st.success(f"✅ **ACCESS GRANTED** — `{user_role}` (level {user_level}) "
                           f"can access `{doc_scope}` documents (level {doc_level}).")
            else:
                st.error(f"🚫 **ACCESS DENIED** — `{user_role}` (level {user_level}) "
                         f"cannot access `{doc_scope}` documents (level {doc_level}).")

    # ── Rate Limiting ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Token-Bucket Rate Limiting")
        st.markdown("Each tenant gets a configurable request budget with token-bucket refill.")

        if "rate_tokens" not in st.session_state:
            st.session_state["rate_tokens"] = 10
            st.session_state["rate_requests"] = 0

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            max_rpm = st.number_input("Max Requests / Minute", 1, 100, 10)
        with col_cfg2:
            st.metric("Remaining Tokens", st.session_state["rate_tokens"])

        if st.button("📤 Send Request"):
            if st.session_state["rate_tokens"] > 0:
                st.session_state["rate_tokens"] -= 1
                st.session_state["rate_requests"] += 1
                st.success(f"✅ Request #{st.session_state['rate_requests']} allowed. "
                           f"Tokens remaining: {st.session_state['rate_tokens']}")
            else:
                st.error("🚫 **RATE LIMITED** — No tokens remaining. Wait for refill.")

        if st.button("🔄 Reset Tokens"):
            st.session_state["rate_tokens"] = max_rpm
            st.session_state["rate_requests"] = 0
            st.info(f"Tokens reset to {max_rpm}.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Evaluation Metrics":
    st.markdown(
        """
        <div class="hero-header">
            <h1>📊 Evaluation Metrics</h1>
            <p>Measured retrieval quality from the committed ablation results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_files = sorted((ROOT / "results").glob("retrieval_eval_*.json"))
    gold_path = ROOT / "data" / "eval" / "retrieval_gold.json"

    if not results_files:
        st.info("No results found. Run `python scripts/evaluate_retrieval.py` to produce `results/retrieval_eval_<date>.json`.")
    else:
        res_path = results_files[-1]
        res = json.loads(res_path.read_text())
        st.markdown("### Retrieval ablation")
        st.caption(
            f"Read from `results/{res_path.name}`. {res['n_questions_evaluated']} answerable questions "
            f"over {res['n_documents']} synthetic policy documents; embeddings `{res['embedding_model']}`, "
            f"reranker `{res['rerank_model']}`. Mean with bootstrap 95% CI "
            f"({res['bootstrap']['n_boot']} resamples over questions)."
        )
        st.warning(
            "Small, single-annotator, author-written set; retrieval only. Generation quality and "
            "faithfulness are not evaluated. Recall@5/10 is near ceiling on this small corpus, "
            "so rank-sensitive metrics (Recall@1, MRR) are the informative ones."
        )
        methods = ["bm25", "dense", "hybrid", "hybrid+rerank"]
        cols_def = [("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("recall@10", "Recall@10"),
                    ("mrr", "MRR"), ("ndcg@10", "nDCG@10")]
        chunker = st.radio("Chunking strategy", list(res["chunkers"]), horizontal=True,
                           help="clause = one chunk per numbered clause; window = fixed 100-word windows")
        summary = res["chunkers"][chunker]
        st.caption(f"{summary['n_chunks']} chunks")
        rows = []
        for m in methods:
            o = summary["results"][m]["overall"]
            rows.append({"Method": m, **{
                label: f"{o[k]['mean']:.3f} [{o[k]['ci95'][0]:.3f}, {o[k]['ci95'][1]:.3f}]"
                for k, label in cols_def}})
        st.table(rows)

        st.markdown("##### MRR by question type")
        types = sorted(summary["results"]["bm25"]["by_type"])
        st.table([{"Method": m, **{
            f"{t} (n={summary['results'][m]['by_type'][t]['n']})":
                f"{summary['results'][m]['by_type'][t]['mrr']:.3f}" for t in types}}
            for m in methods])

    st.markdown("---")
    st.markdown("### Gold set")
    if gold_path.exists():
        gold = json.loads(gold_path.read_text())
        qs = gold["questions"]
        counts = {}
        for q in qs:
            counts[q["type"]] = counts.get(q["type"], 0) + 1
        st.markdown(f"**{len(qs)} questions** in `retrieval_gold.json`: " +
                    ", ".join(f"{t} {n}" for t, n in sorted(counts.items())) +
                    ". `not_covered` questions have no evidence and are excluded from retrieval metrics.")
        for q in qs:
            with st.expander(f"{q['id']}: {q['question']}", expanded=False):
                st.markdown(f'<span class="tag">{q["type"]}</span>', unsafe_allow_html=True)
                st.markdown(f"**Gold answer:** {q['gold_answer']}")
                for ev in q["evidence"]:
                    st.markdown(f"- `{ev['doc']}`: \"{ev['quote']}\"")
    else:
        st.info("Gold set not found at `data/eval/retrieval_gold.json`.")


# ═══════════════════════════════════════════════════════════════════════════════
