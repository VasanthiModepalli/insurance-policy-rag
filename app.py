"""Streamlit UI. Run with:  streamlit run app.py"""

import os
import tempfile

import streamlit as st

from rag_core import answer, build_index, load_chunks

DEFAULT_PDF = "policy.pdf"  # shown automatically if nothing is uploaded

st.set_page_config(page_title="Ask My PDF", page_icon="📄")
st.title("📄 Ask My PDF")
st.caption("Answers come only from the document, with page citations.")

chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 1000, 500, step=100)
top_k = st.sidebar.slider("Chunks retrieved (top-k)", 1, 8, 4)

uploaded = st.file_uploader("Upload a PDF", type="pdf")

pdf_path, pdf_name = None, None
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded.getvalue())
    tmp.close()
    pdf_path, pdf_name = tmp.name, uploaded.name
elif os.path.exists(DEFAULT_PDF):
    pdf_path, pdf_name = DEFAULT_PDF, DEFAULT_PDF
    st.info(f"Using the built-in sample document: {DEFAULT_PDF}")

if pdf_path is None:
    st.warning("Upload a PDF to get started.")
    st.stop()

# Rebuild the index only when the file or chunk size changes
index_key = (pdf_name, chunk_size)
if st.session_state.get("index_key") != index_key:
    with st.spinner("Reading and indexing the document..."):
        chunks = load_chunks(pdf_path, chunk_size=chunk_size, overlap=50)
        st.session_state["collection"] = build_index(chunks)
        st.session_state["n_chunks"] = len(chunks)
        st.session_state["index_key"] = index_key

st.write(f"Indexed **{st.session_state['n_chunks']}** chunks.")

question = st.text_input("Ask a question about the document")
if question:
    with st.spinner("Thinking..."):
        text, sources = answer(question, st.session_state["collection"], top_k=top_k)
    st.markdown("### Answer")
    st.write(text)

    with st.expander("Sources used"):
        for s in sources:
            st.markdown(f"**Page {s['page']}** (similarity {s['similarity']})")
            st.caption(s["text"])
