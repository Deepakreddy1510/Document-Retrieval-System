from __future__ import annotations

from typing import Any

import psycopg
import streamlit as st
from psycopg.rows import dict_row

from src.pdf_rag.config import settings
from src.pdf_rag.pipeline import RAGPipeline


ABSTAIN_MESSAGE = (
    "I couldn't find enough evidence in the uploaded documents to answer that."
)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Small, safe styling only
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --rag-bg: #0d1117;
        --rag-panel: #141922;
        --rag-panel-2: #191f2a;
        --rag-border: #2b3240;
        --rag-text: #f3f4f6;
        --rag-muted: #9aa4b2;
        --rag-accent: #7c6cff;
    }

    .stApp {
        background: var(--rag-bg);
        color: var(--rag-text);
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    header[data-testid="stHeader"] {
        background: rgba(13, 17, 23, 0.92);
        border-bottom: 1px solid var(--rag-border);
    }

    [data-testid="stSidebar"] {
        background: #10151d;
        border-right: 1px solid var(--rag-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--rag-text);
    }

    [data-testid="stMetric"] {
        background: var(--rag-panel);
        border: 1px solid var(--rag-border);
        border-radius: 12px;
        padding: 0.7rem 0.8rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--rag-panel);
        border: 1px dashed #3a4352;
        border-radius: 12px;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
    }

    [data-testid="stChatMessageContent"] {
        color: var(--rag-text);
    }

    [data-testid="stChatInput"] {
        background: rgba(13, 17, 23, 0.96);
        border-top: 1px solid var(--rag-border);
    }

    [data-testid="stChatInput"] textarea {
        background: var(--rag-panel) !important;
        color: var(--rag-text) !important;
        border: 1px solid var(--rag-border) !important;
        border-radius: 14px !important;
    }

    [data-testid="stExpander"] {
        background: var(--rag-panel);
        border: 1px solid var(--rag-border) !important;
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary {
        color: var(--rag-text);
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--rag-border);
        background: var(--rag-panel-2);
        color: var(--rag-text);
    }

    .stButton > button:hover {
        border-color: var(--rag-accent);
        color: #ffffff;
    }

    .stButton > button[kind="primary"] {
        background: var(--rag-accent);
        border-color: var(--rag-accent);
        color: white;
    }

    [data-testid="stAlert"] {
        background: var(--rag-panel);
        border: 1px solid var(--rag-border);
        color: var(--rag-text);
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--rag-muted) !important;
    }

    code {
        background: #0b0f14 !important;
        color: #d8dee9 !important;
    }

    hr {
        border-color: var(--rag-border) !important;
    }

    #MainMenu,
    footer,
    [data-testid="stDeployButton"] {
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


pipeline = get_pipeline()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_value(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a value from either a dict or an object/dataclass."""
    if obj is None:
        return default

    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]

        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def normalize_sources(raw_sources: Any) -> list[dict]:
    """Convert source objects to plain dictionaries for session state."""
    sources: list[dict] = []

    for i, source in enumerate(raw_sources or [], start=1):
        label = get_value(source, "label", "citation_id", default=f"S{i}")
        filename = get_value(
            source,
            "filename",
            "document",
            "document_name",
            default="Unknown document",
        )
        page_number = get_value(
            source,
            "page_number",
            "page",
            default=None,
        )
        text = get_value(
            source,
            "text",
            "content",
            "excerpt",
            "passage",
            default="",
        )

        sources.append(
            {
                "label": str(label),
                "filename": str(filename),
                "page_number": page_number,
                "text": str(text or ""),
            }
        )

    return sources


def is_abstention(answer: str) -> bool:
    return answer.strip() == ABSTAIN_MESSAGE


def load_documents() -> list[dict]:
    """Load document counts for the sidebar."""
    query = """
        SELECT
            d.id,
            d.filename,
            d.page_count,
            COUNT(c.id)::int AS chunk_count
        FROM documents AS d
        LEFT JOIN chunks AS c
            ON c.document_id = d.id
        GROUP BY d.id, d.filename, d.page_count
        ORDER BY d.id DESC;
    """

    try:
        with psycopg.connect(
            settings.database_url,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        st.sidebar.error(f"Could not read document statistics: {exc}")
        return []


def render_evidence(sources: list[dict]) -> None:
    """Show retrieved evidence only for supported answers."""
    if not sources:
        return

    with st.expander(f"Evidence ({len(sources)} passages)"):
        for index, source in enumerate(sources):
            label = source.get("label", f"S{index + 1}")
            filename = source.get("filename", "Unknown document")
            page = source.get("page_number")
            passage = source.get("text", "").strip()

            if page is None:
                st.markdown(f"**[{label}] {filename}**")
            else:
                st.markdown(f"**[{label}] {filename} — page {page}**")

            if passage:
                st.caption(passage)

            if index < len(sources) - 1:
                st.divider()


def render_message(message: dict) -> None:
    """Render one chat message using Streamlit's default avatars."""
    role = message.get("role", "assistant")
    content = message.get("content", "")

    # Do not pass a custom avatar.
    # Streamlit's default user/assistant icons are reliable across versions.
    with st.chat_message(role):
        st.markdown(content)

        if (
            role == "assistant"
            and not is_abstention(content)
            and message.get("sources")
        ):
            render_evidence(message["sources"])


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
documents = load_documents()
document_count = len(documents)
chunk_count = sum(int(doc.get("chunk_count", 0) or 0) for doc in documents)

with st.sidebar:
    st.header("Documents")
    st.caption("Upload and index text-based PDF files.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Index selected PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        with st.spinner("Indexing PDFs..."):
            for uploaded_file in uploaded_files:
                try:
                    result = pipeline.ingest_pdf(
                        filename=uploaded_file.name,
                        pdf_bytes=uploaded_file.getvalue(),
                    )

                    if result.status == "already_indexed":
                        st.info(f"{result.filename} is already indexed.")
                    else:
                        st.success(
                            f"Indexed {result.filename}: "
                            f"{result.page_count} pages, "
                            f"{result.chunk_count} chunks."
                        )

                except Exception as exc:
                    st.error(f"{uploaded_file.name}: {exc}")

        st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Documents", document_count)
    col2.metric("Chunks", f"{chunk_count:,}")

    if documents:
        st.subheader("Indexed PDFs")

        for document in documents:
            pages = int(document.get("page_count", 0) or 0)
            chunks = int(document.get("chunk_count", 0) or 0)

            with st.container(border=True):
                st.markdown(f"**📄 {document['filename']}**")
                st.caption(f"{pages:,} pages · {chunks:,} chunks")
    else:
        st.caption("No PDFs indexed yet.")

    st.divider()

    with st.expander("Technical details"):
        st.markdown("**Production retrieval**")
        st.code("Dense retrieval → CrossEncoder reranking → Top-K", language=None)

        st.markdown("**Embedding model**")
        st.code(settings.embedding_model, language=None)

        st.markdown("**Reranker**")
        st.code(settings.reranker_model, language=None)

        st.markdown("**LLM**")
        st.code(
            f"{settings.llm_provider} / {settings.llm_model}",
            language=None,
        )

    if st.session_state.messages:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
st.title("📄 PDF RAG Assistant")
st.caption(
    "Ask questions across your indexed PDFs. "
    "Supported answers include page-level evidence."
)

if document_count == 0:
    st.info("Upload and index at least one PDF from the sidebar to begin.")

if not st.session_state.messages and document_count > 0:
    st.markdown("### Ask your first question")
    st.write(
        "The assistant retrieves relevant passages from your PDFs, "
        "reranks them, and answers using only the retrieved evidence."
    )

    suggestion_cols = st.columns(3)

    suggestions = [
        "Summarize the main ideas in the uploaded documents.",
        "Explain one important concept from the uploaded documents.",
        "What are the major differences between the uploaded documents?",
    ]

    for column, suggestion in zip(suggestion_cols, suggestions):
        with column:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion


# Render chat history
for message in st.session_state.messages:
    render_message(message)


# ---------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------
pending_question = st.session_state.pop("pending_question", None)

typed_question = st.chat_input(
    "Ask a question about your indexed PDFs",
    disabled=document_count == 0,
)

question = pending_question or typed_question

if question and question.strip():
    question = question.strip()

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    # Use default Streamlit avatar - no custom avatar argument.
    with st.chat_message("user"):
        st.markdown(question)

    # Use default Streamlit assistant avatar - no custom avatar argument.
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs..."):
            try:
                result = pipeline.answer(question)

                answer_text = result.answer.strip()
                sources = normalize_sources(result.sources)

                st.markdown(answer_text)

                # If the assistant refuses because evidence is insufficient,
                # do not show retrieved candidates as "evidence".
                if not is_abstention(answer_text):
                    render_evidence(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": (
                            []
                            if is_abstention(answer_text)
                            else sources
                        ),
                    }
                )

            except Exception as exc:
                error_text = f"Something went wrong while answering: {exc}"
                st.error(error_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_text,
                        "sources": [],
                    }
                )

    st.rerun()
