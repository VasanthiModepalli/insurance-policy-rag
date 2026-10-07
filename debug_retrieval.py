"""
See what the retriever finds for a question. Uses NO LLM, so it costs no quota.

Run:
    python debug_retrieval.py
    python debug_retrieval.py "your own question here"
"""

import sys

from rag_core import build_index, get_embedder, load_chunks

PDF_PATH = "policy.pdf"
question = " ".join(sys.argv[1:]) or "What do the benefits together give?"

for size in (300, 800):
    chunks = load_chunks(PDF_PATH, chunk_size=size, overlap=50)
    collection = build_index(chunks)
    q_emb = get_embedder().encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=4)

    print("\n" + "=" * 70)
    print(f"chunk_size={size} ({len(chunks)} chunks)  |  Q: {question}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n  page {meta['page']} | similarity {1 - dist:.2f}")
        print("  " + doc)