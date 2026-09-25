import hashlib
import os

import streamlit as st

from utils.pdf_loader import extract_pages_from_pdf
from utils.text_splitter import split_pages
from utils.vector_store import VectorStore
from utils.llm import stream_answer, DEFAULT_MODEL

st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="wide")

defaults = {
    "messages": [],
    "vector_store": None,
    "indexed_hash": None,
    "indexed_files": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def files_hash(files, settings) -> str:
    """Fingerprint documents + retrieval settings so stale indexes are never reused."""
    h = hashlib.sha256()
    h.update(repr(sorted(settings.items())).encode("utf-8"))
    for f in files:
        f.seek(0)
        h.update(f.read())
        f.seek(0)
    return h.hexdigest()


with st.sidebar:
    st.header("📄 AI Support Agent")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    with st.expander("⚙️ Advanced RAG settings", expanded=True):
        chunk_size = st.slider("Chunk size (characters)", 400, 1800, 900, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, step=20)
        top_k = st.slider("Final chunks", 1, 8, 5)
        candidate_k = st.slider("Candidate pool", 10, 60, 30, step=5)
        use_reranker = st.checkbox(
            "Cross-encoder reranking", value=True,
            help="Runs a second relevance model over hybrid retrieval candidates."
        )
        mmr_lambda = st.slider(
            "Diversity / relevance", 0.50, 0.95, 0.72, step=0.01,
            help="Higher values prioritize relevance; lower values diversify the final context."
        )
        min_rerank_score = st.slider(
            "Minimum reranker score", -3.0, 2.0, -1.5, step=0.1,
            help="Raises the relevance gate when you want fewer, stricter sources."
        )
        temperature = st.slider("Response creativity", 0.0, 1.0, 0.2, step=0.1)
        model = st.selectbox(
            "Model",
            [DEFAULT_MODEL, "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("♻️ Reset Index", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.indexed_hash = None
            st.session_state.indexed_files = []
            st.rerun()

    if st.session_state.messages:
        chat_txt = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Download chat",
            chat_txt,
            file_name="chat_history.txt",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Advanced RAG pipeline")
    st.markdown(
        "- Dense FAISS retrieval\n"
        "- BM25 keyword retrieval\n"
        "- Reciprocal Rank Fusion\n"
        "- Cross-encoder reranking\n"
        "- Duplicate filtering + MMR\n"
        "- Grounded source/page citations"
    )

st.title("🤖 AI Support Agent")
st.caption("Multi-stage hybrid RAG: retrieve → fuse → rerank → diversify → generate")

settings = {
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,
    "top_k": top_k,
    "candidate_k": candidate_k,
    "use_reranker": use_reranker,
    "mmr_lambda": mmr_lambda,
    "min_rerank_score": min_rerank_score,
}

if uploaded_files:
    current_hash = files_hash(uploaded_files, settings)

    if current_hash != st.session_state.indexed_hash:
        try:
            all_chunks = []
            progress = st.progress(0.0, text="Reading and cleaning PDFs…")

            for i, file in enumerate(uploaded_files):
                pages = extract_pages_from_pdf(file)
                chunks = split_pages(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    source=file.name,
                )
                all_chunks.extend(chunks)
                progress.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"Processed {file.name}",
                )

            progress.progress(1.0, text="Building hybrid retrieval indexes…")
            store = VectorStore()
            store.create_index(all_chunks)

            st.session_state.vector_store = store
            st.session_state.indexed_hash = current_hash
            st.session_state.indexed_files = [f.name for f in uploaded_files]
            progress.empty()
            st.toast(
                f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s) with hybrid RAG ✅"
            )
        except Exception as e:
            st.error(f"Couldn't process your PDF(s): {e}")
            st.stop()
    else:
        st.caption(
            f"✅ Cached index: {', '.join(st.session_state.indexed_files)}"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.vector_store:
    question = st.chat_input("Ask anything about your PDF(s)...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Running advanced retrieval…"):
                try:
                    results = st.session_state.vector_store.search(
                        question,
                        k=top_k,
                        candidate_k=candidate_k,
                        rerank=use_reranker,
                        mmr_lambda=mmr_lambda,
                        min_rerank_score=min_rerank_score,
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.stop()

            if not results:
                answer = "I couldn't find sufficiently relevant information in the uploaded document(s)."
                st.markdown(answer)
            else:
                context_parts = []
                for i, r in enumerate(results, start=1):
                    source = r.source or "document"
                    page = f", page {r.page_number}" if r.page_number else ""
                    context_parts.append(
                        f"[SOURCE {i} | {source}{page}]\n{r.text}"
                    )
                context = "\n\n".join(context_parts)

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                placeholder = st.empty()
                answer = ""
                try:
                    for token in stream_answer(
                        context,
                        question,
                        model=model,
                        temperature=temperature,
                        history=history,
                    ):
                        answer += token
                        placeholder.markdown(answer + "▌")
                    placeholder.markdown(answer)
                except Exception as e:
                    answer = f"⚠️ Something went wrong: {e}"
                    placeholder.markdown(answer)

                with st.expander("📄 Retrieval Trace"):
                    st.caption(
                        f"Hybrid candidates: up to {candidate_k} · Final context: {len(results)} chunks"
                    )
                    for i, r in enumerate(results, start=1):
                        page_info = f" — page {r.page_number}" if r.page_number else ""
                        source_info = f" ({r.source})" if r.source else ""
                        rerank_info = (
                            f" · reranker {r.rerank_score:.2f}"
                            if r.rerank_score is not None
                            else ""
                        )
                        st.markdown(
                            f"**Source {i}{source_info}{page_info}** · "
                            f"retrieval {r.retrieval_score:.3f}{rerank_info}"
                        )
                        st.write(r.text)
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload one or more PDFs from the sidebar to begin.")
