from rag_core import load_chunks

chunks = load_chunks("policy.pdf")
print("Total chunks:", len(chunks))
for c in chunks[:5]:
    print(c["page"], "|", c["text"][:150])
    print("-" * 40)