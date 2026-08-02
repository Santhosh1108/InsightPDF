import hashlib
import os

import streamlit as st

from utils.pdf_loader import extract_pages_from_pdf
from utils.text_splitter import split_pages
from utils.vector_store import VectorStore
from utils.llm import stream_answer, DEFAULT_MODEL

st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="wide")

# ---------- Session state ----------
defaults = {
    "messages": [],
    "vector_store": None,
    "indexed_hash": None,
    "indexed_files": [],
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def files_hash(files) -> str:
    """Fingerprint the uploaded set so we only rebuild the index when the
    files actually change, instead of on every single chat message.
    """
    h = hashlib.sha256()
    for f in files:
        f.seek(0)
        h.update(f.read())
        f.seek(0)
    return h.hexdigest()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 AI Support Agent")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    

    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
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
    st.subheader("Tech Stack")
    st.markdown("- Streamlit\n- FAISS\n- Sentence Transformers\n- Groq\n- Llama 3")

st.title("🤖 AI Support Agent")

# ---------- Indexing (only rebuilds when files actually change) ----------
if uploaded_files:
    current_hash = files_hash(uploaded_files)

    if current_hash != st.session_state.indexed_hash:
        try:
            all_chunks = []
            progress = st.progress(0.0, text="Reading PDFs…")
            for i, file in enumerate(uploaded_files):
                pages = extract_pages_from_pdf(file)
                chunks = split_pages(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    source=file.name,
                )
                all_chunks.extend(chunks)
                progress.progress((i + 1) / len(uploaded_files), text=f"Processed {file.name}")

            progress.progress(1.0, text="Building search index…")
            store = VectorStore()
            store.create_index(all_chunks)

            st.session_state.vector_store = store
            st.session_state.indexed_hash = current_hash
            st.session_state.indexed_files = [f.name for f in uploaded_files]
            progress.empty()
            st.toast(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s) ✅")
        except Exception as e:
            st.error(f"Couldn't process your PDF(s): {e}")
            st.stop()
    else:
        st.caption(f"✅ Using cached index for: {', '.join(st.session_state.indexed_files)}")

# ---------- Chat history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Chat input ----------
if st.session_state.vector_store:
    question = st.chat_input("Ask anything about your PDF(s)...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document…"):
                try:
                    results = st.session_state.vector_store.search(question, k=top_k)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.stop()

            if not results:
                answer = "I couldn't find anything relevant to that in the document."
                st.markdown(answer)
            else:
                context = "\n\n".join(r.text for r in results)
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

                with st.expander("📄 Sources Used"):
                    for i, r in enumerate(results, start=1):
                        page_info = f" — page {r.page_number}" if r.page_number else ""
                        source_info = f" ({r.source})" if r.source else ""
                        st.markdown(
                            f"**Source {i}{source_info}{page_info}** · relevance {r.score:.2f}"
                        )
                        st.write(r.text)
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload one or more PDFs from the sidebar to begin.")
