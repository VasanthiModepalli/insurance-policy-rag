"""Core RAG logic: load PDF -> chunk -> embed/store -> retrieve -> ask LLM."""

import os
import time
import unicodedata
import uuid

import chromadb
from google import genai
from google.genai import errors
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# If this model name errors out, open Google AI Studio, copy a current
# model name from there (e.g. a "flash" model) and paste it here.
LLM_MODEL = "gemini-3.5-flash"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def load_chunks(pdf_path, chunk_size=500, overlap=50):
    """Read the PDF page by page and cut it into overlapping chunks."""
    reader = PdfReader(pdf_path)
    step = chunk_size - overlap
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = unicodedata.normalize("NFKC", page.extract_text() or "")  # fixes ligatures like \ufb01 -> fi
        text = text.replace("\n", " ")
        for i in range(0, len(text), step):
            piece = text[i:i + chunk_size].strip()
            if len(piece) > 50:  # skip tiny fragments
                chunks.append({"text": piece, "page": page_num})
    return chunks


def build_index(chunks):
    """Embed all chunks and store them in an in-memory ChromaDB collection."""
    client = chromadb.Client()
    collection = client.create_collection(
        name=f"docs_{uuid.uuid4().hex[:8]}",
        metadata={"hnsw:space": "cosine"},
    )
    texts = [c["text"] for c in chunks]
    embeddings = get_embedder().encode(texts, show_progress_bar=False)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return collection


def answer(question, collection, top_k=4):
    """Retrieve top_k chunks, ask the LLM, return (answer_text, sources)."""
    q_emb = get_embedder().encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    context = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in zip(docs, metas))
    prompt = f"""You are a helpful assistant answering questions about a document.
Answer using ONLY the context below.
If the answer is not in the context, reply exactly: "I couldn't find this in the document."
Mention the page numbers you used.

Context:
{context}

Question: {question}"""

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # Retry when Google's servers are busy (503) or the free-tier limit is hit (429)
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0},
            )
            break
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"  [API busy (error {e.code}), retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise
    text = resp.text

    sources = [
        {"page": m["page"], "text": d, "similarity": round(1 - dist, 2)}
        for d, m, dist in zip(docs, metas, dists)
    ]
    return text, sources
