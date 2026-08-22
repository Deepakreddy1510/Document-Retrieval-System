from __future__ import annotations

import html
from typing import Any

import psycopg
import streamlit as st
from psycopg.rows import dict_row

from src.pdf_rag.config import settings
from src.pdf_rag.pipeline import RAGPipeline


ABSTAIN_MESSAGE = (
    "I couldn't find enough evidence in the uploaded documents to answer that."
)


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --rag-border: color-mix(in srgb, var(--text-color) 14%, transparent);
            --rag-muted: color-mix(in srgb, var(--text-color) 66%, transparent);
            --rag-soft: color-mix(in srgb, var(--secondary-background-color) 88%, transparent);
        }

        .stApp { background: var(--background-color); }

        .block-container {
            max-width: 1080px;
            padding-top: 2.3rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--rag-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        [data-testid="stDeployButton"] { display: none; }

        .rag-header { margin-bottom: 1.6rem; }

        .rag-title-row {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .rag-logo {
            width: 46px;
            height: 46px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            background: color-mix(in srgb, var(--primary-color) 16%, transparent);
            border: 1px solid color-mix(in srgb, var(--primary-color) 30%, transparent);
            font-size: 24px;
        }

        .rag-title {
            margin: 0;
            font-size: 2.15rem;
            font-weight: 760;
            letter-spacing: -0.035em;
            line-height: 1.1;
        }

        .rag-subtitle {
            margin-top: 0.65rem;
            color: var(--rag-muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .rag-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }

        .rag-badge {
            padding: 0.34rem 0.65rem;
            border-radius: 999px;
            border: 1px solid var(--rag-border);
            background: var(--rag-soft);
            color: var(--rag-muted);
            font-size: 0.82rem;
        }

        .rag-empty {
            border: 1px solid var(--rag-border);
            border-radius: 18px;
            padding: 2.2rem;
            background: var(--rag-soft);
            margin-top: 1.1rem;
        }

        .rag-empty-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .rag-empty-text {
            color: var(--rag-muted);
            line-height: 1.6;
            margin: 0;
        }

        .doc-card {
            padding: 0.78rem 0.85rem;
            border: 1px solid var(--rag-border);
            border-radius: 13px;
            margin-bottom: 0.6rem;
            background: var(--rag-soft);
        }

        .doc-name {
            font-weight: 650;
            line-height: 1.35;
            word-break: break-word;
        }

        .doc-meta {
            margin-top: 0.28rem;
            color: var(--rag-muted);
            font-size: 0.82rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid var(--rag-border);
            border-radius: 13px;
            padding: 0.7rem 0.8rem;
            background: var(--rag-soft);
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--rag-border);
            border-radius: 17px;
            padding: 0.35rem 0.45rem;
            margin-bottom: 0.75rem;
            background: color-mix(
                in srgb,
                var(--secondary-background-color) 72%,
                transparent
            );
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid var(--rag-border);
            padding-top: 0.75rem;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--rag-border);
            border-radius: 13px;
            overflow: hidden;
        }

        .evidence-label {
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .evidence-meta {
            color: var(--rag-muted);
            font-size: 0.85rem;
            margin-bottom: 0.45rem;
        }

        @media (max-width: 800px) {
            .block-container { padding-top: 1.25rem; }
            .rag-title { font-size: 1.75rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


pipeline = get_pipeline()


def get_value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def normalize_sources(raw_sources: Any) -> list[dict]:
    normalized: list[dict] = []
    for index, source in enumerate(raw_sources or [], start=1):
        label = get_value(source, "label", "citation_id", default=f"S{index}")
        label = str(label)
        if not label.startswith("S"):
            label = f"S{index}"

        normalized.append(
            {
                "label": label,
                "filename": str(
                    get_value(
                        source,
                        "filename",
                        "document",
                        "document_name",
                        default="Unknown document",
                    )
                ),
                "page_number": get_value(source, "page_number", "page", default=None),
                "text": str(
                    get_value(
                        source,
                        "text",
                        "content",
                        "excerpt",
                        "passage",
                        default="",
                    )
                    or ""
                ),
            }
        )
    return normalized


def is_abstention(answer: str) -> bool:
    return answer.strip() == ABSTAIN_MESSAGE


def load_documents() -> list[dict]:
    query = """
        SELECT
            d.id,
            d.filename,
            d.page_count,
            COUNT(c.id)::int AS chunk_count
        FROM documents AS d
        LEFT JOIN chunks AS c
            ON c.document_id = d.id
        GROUP BY
            d.id,
            d.filename,
            d.page_count
        ORDER BY d.id DESC;
    """

    try:
        with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        st.sidebar.error(f"Could not load document stats: {exc}")
        return []


def render_evidence(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"Evidence · {len(sources)} passages"):
        for index, source in enumerate(sources):
            label = source.get("label", f"S{index + 1}")
            filename = source.get("filename", "Unknown document")
            page = source.get("page_number")
            passage = source.get("text", "").strip()
            page_text = f"page {page}" if page is not None else "page unavailable"

            st.markdown(
                f"""
                <div class="evidence-label">[{html.escape(str(label))}] {html.escape(str(filename))}</div>
                <div class="evidence-meta">{html.escape(page_text)}</div>
                """,
                unsafe_allow_html=True,
            )

            if passage:
                st.caption(passage)

            if index < len(sources) - 1:
                st.divider()


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    avatar = "👤" if role == "user" else "✦"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if (
            role == "assistant"
            and not is_abstention(content)
            and message.get("sources")
        ):
            render_evidence(message["sources"])


if "messages" not in st.session_state:
    st.session_state.messages = []


documents = load_documents()
document_count = len(documents)
chunk_count = sum(int(doc.get("chunk_count", 0) or 0) for doc in documents)


with st.sidebar:
    st.markdown("## Documents")
    st.caption("Upload text-based PDFs and index them for retrieval.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDF files.",
    )

    process_clicked = st.button(
        "Index selected PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_clicked and uploaded_files:
        statuses = []

        with st.spinner("Extracting, chunking, and embedding PDFs..."):
            for uploaded_file in uploaded_files:
                try:
                    result = pipeline.ingest_pdf(
                        filename=uploaded_file.name,
                        pdf_bytes=uploaded_file.getvalue(),
                    )
                    statuses.append(result)
                except Exception as exc:
                    st.error(f"{uploaded_file.name}: {exc}")

        for result in statuses:
            if result.status == "already_indexed":
                st.info(f"{result.filename} is already indexed.")
            else:
                st.success(
                    f"Indexed {result.filename} · "
                    f"{result.page_count} pages · "
                    f"{result.chunk_count} chunks"
                )

        st.rerun()

    st.divider()

    left, right = st.columns(2)
    with left:
        st.metric("Documents", document_count)
    with right:
        st.metric("Chunks", f"{chunk_count:,}")

    if documents:
        st.markdown("#### Indexed PDFs")
        for document in documents:
            filename = html.escape(str(document["filename"]))
            pages = int(document.get("page_count", 0) or 0)
            chunks = int(document.get("chunk_count", 0) or 0)

            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-name">📄 {filename}</div>
                    <div class="doc-meta">{pages:,} pages · {chunks:,} chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No PDFs indexed yet.")

    st.divider()

    with st.expander("Technical details"):
        st.caption("Production retrieval")
        st.code("Dense Top-N → CrossEncoder → Final Top-K", language=None)

        st.caption("Embedding model")
        st.code(settings.embedding_model, language=None)

        st.caption("Reranker")
        st.code(settings.reranker_model, language=None)

        st.caption("LLM")
        st.code(f"{settings.llm_provider} / {settings.llm_model}", language=None)

        st.caption(
            "Lexical retrieval and RRF remain in the codebase as evaluated "
            "experimental configurations."
        )

    if st.session_state.messages:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


plural = "s" if document_count != 1 else ""
st.markdown(
    f"""
    <div class="rag-header">
        <div class="rag-title-row">
            <div class="rag-logo">📄</div>
            <h1 class="rag-title">PDF RAG Assistant</h1>
        </div>

        <div class="rag-subtitle">
            Ask questions across your indexed PDFs. Answers are grounded in retrieved
            passages with page-level evidence.
        </div>

        <div class="rag-badges">
            <span class="rag-badge">{document_count} indexed document{plural}</span>
            <span class="rag-badge">{chunk_count:,} searchable chunks</span>
            <span class="rag-badge">Dense retrieval + reranking</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.messages:
    if document_count == 0:
        empty_title = "Upload a PDF to get started"
        empty_text = (
            "Use the sidebar to upload one or more text-based PDF files. After indexing, "
            "you can ask questions and inspect the evidence behind each supported answer."
        )
    else:
        empty_title = "Your documents are ready"
        empty_text = (
            "Ask a question below. The assistant will retrieve relevant passages, rerank "
            "them, and answer only from the available evidence."
        )

    st.markdown(
        f"""
        <div class="rag-empty">
            <div class="rag-empty-title">{empty_title}</div>
            <p class="rag-empty-text">{empty_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    render_message(message)


question = st.chat_input(
    "Ask a question about your indexed PDFs",
    disabled=document_count == 0,
)

if question:
    question = question.strip()

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with st.chat_message("assistant", avatar="✦"):
            with st.spinner("Searching your documents..."):
                try:
                    result = pipeline.answer(question)
                    answer_text = result.answer.strip()
                    sources = normalize_sources(result.sources)

                    st.markdown(answer_text)

                    # Hide nearest-neighbour candidates when the system abstains.
                    # They were insufficient evidence, not sources supporting an answer.
                    if not is_abstention(answer_text):
                        render_evidence(sources)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": [] if is_abstention(answer_text) else sources,
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
