"""Core RAG logic: load PDF -> chunk -> embed/store -> retrieve -> ask LLM.

LLM provider is chosen with environment variables (no code changes needed):

  Gemini (default):  set GEMINI_API_KEY=...            [set LLM_MODEL=...]
  OpenRouter:        set LLM_PROVIDER=openrouter
                     set OPENROUTER_API_KEY=...
                     set LLM_MODEL=<a model name ending in :free>
  Ollama (local):    set LLM_PROVIDER=ollama           [set LLM_MODEL=llama3.2]
"""

import hashlib
import json
import os
import time
import unicodedata
import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

_DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "openrouter": "",           # you must choose one, see openrouter.ai/models
    "ollama": "llama3.2",
}
LLM_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODELS.get(PROVIDER, ""))

# Answers are cached on disk, so re-running tests never wastes free-tier quota.
CACHE_FILE = "llm_cache.json"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------- PDF + index
def load_chunks(pdf_path, chunk_size=500, overlap=50):
    """Read the PDF page by page and cut it into overlapping chunks."""
    reader = PdfReader(pdf_path)
    step = chunk_size - overlap
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = unicodedata.normalize("NFKC", page.extract_text() or "")  # fixes ligatures (fi)
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


# ------------------------------------------------------------------- LLM call
def _load_cache():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def _call_llm_once(prompt):
    """One request to whichever provider is selected."""
    if PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model=LLM_MODEL, contents=prompt, config={"temperature": 0}
        )
        return resp.text

    from openai import OpenAI  # used for both OpenRouter and Ollama

    if PROVIDER == "openrouter":
        if not LLM_MODEL:
            raise RuntimeError(
                "Set LLM_MODEL to an OpenRouter model name (pick one ending in :free "
                "from openrouter.ai/models)."
            )
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    elif PROVIDER == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{PROVIDER}'. Use gemini, openrouter or ollama.")

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def _status_code(err):
    return getattr(err, "code", None) or getattr(err, "status_code", None)


def call_llm(prompt):
    """Cached + retrying LLM call."""
    cache = _load_cache()
    key = hashlib.sha256(f"{PROVIDER}|{LLM_MODEL}|{prompt}".encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    for attempt in range(6):
        try:
            text = _call_llm_once(prompt)
            break
        except Exception as e:  # noqa: BLE001
            code = _status_code(e)
            if code == 429 and "PerDay" in str(e):
                raise RuntimeError(
                    f"Daily free-tier quota for '{LLM_MODEL}' is used up. Switch model or "
                    "provider (see top of rag_core.py) or try again tomorrow. Answers "
                    "already received are saved in llm_cache.json."
                ) from None
            if code in (429, 500, 502, 503) and attempt < 5:
                wait = 10 * (attempt + 1)
                print(f"  [LLM busy (error {code}), retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise

    cache[key] = text
    _save_cache(cache)
    return text


# --------------------------------------------------------------------- answer
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

    text = call_llm(prompt)

    sources = [
        {"page": m["page"], "text": d, "similarity": round(1 - dist, 2)}
        for d, m, dist in zip(docs, metas, dists)
    ]
    return text, sources
