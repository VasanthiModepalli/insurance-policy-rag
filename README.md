# insurance-policy-rag

📄 Ask My PDF: RAG Q&A over an Insurance Policy

A question-answering app that answers only from a given document and cites the page numbers it used. Built on a public health insurance brochure (HDFC Ergo Optima Secure), but it works with any text-based PDF.

Live demo: add your Streamlit / Hugging Face link here Demo GIF: add a short screen recording here

Why this project?

A general-purpose LLM doesn't know the details of a specific policy, and if you ask it anyway it may guess. This project uses Retrieval-Augmented Generation (RAG) so the model answers from the document itself:

Answers are grounded in retrieved text, not the model's memory.
Every answer shows the page numbers it came from, so it can be verified.
If the answer isn't in the document, the model is instructed to say "I couldn't find this in the document." instead of making something up.
How it works
PDF
 └─► extract text page by page (pypdf)
      └─► split into overlapping chunks (500 chars, 50 overlap), keeping page numbers
           └─► embed each chunk (sentence-transformers: all-MiniLM-L6-v2)
                └─► store vectors in ChromaDB

Question
 └─► embed the question
      └─► retrieve the top-k most similar chunks from ChromaDB
           └─► build a prompt: "answer ONLY from this context, cite pages"
                └─► Gemini generates the answer
                     └─► show answer + sources (page, similarity score)
Tech stack
Part	Tool
PDF reading	pypdf
Embeddings	sentence-transformers (all-MiniLM-L6-v2, runs locally)
Vector store	ChromaDB (in-memory, cosine similarity)
LLM	Google Gemini API (free tier)
UI	Streamlit
Project structure
├── rag_core.py        # chunking, indexing, retrieval, LLM call (with retry + cache)
├── app.py             # Streamlit UI
├── run_tests.py       # runs test questions at different chunk sizes, saves a CSV
├── check_pdf.py       # quick check that the PDF text extracts properly
├── requirements.txt
└── policy.pdf         # the document (public brochure)
Run it locally

1. Clone and install

bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

2. Get a free Gemini API key from Google AI Studio and set it as an environment variable:

bash
# Windows (cmd)
set GEMINI_API_KEY=your_key_here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your_key_here"

# Mac / Linux
export GEMINI_API_KEY=your_key_here

The key is read from the environment and is never stored in the code.

3. Start the app

bash
streamlit run app.py

Upload any PDF, or use the built-in policy.pdf. The sidebar lets you change the chunk size and the number of retrieved chunks (top-k).

Optional: switch the model without editing code:

bash
set LLM_MODEL=your-model-name
Example output

Real answers observed while testing:

Question	Answer (abridged)
Is there an extra charge for the restored amount?	No extra charge; the restored cover is available for later claims "at no additional cost" (Page 5).
How many times can cover be restored in a year?	Once per policy year (Page 5); with the "Unlimited Restore" add-on, unlimited (Page 15).
Evaluation

I tested the system with a fixed set of questions, including questions whose answers are not in the document, to check for hallucination. Run it yourself with:

bash
python run_tests.py

This writes a new CSV (results_<model>_<timestamp>.csv) with each question, the expected answer, the model's answer and the pages cited. The verdict column is filled in by hand.

Results (fill in after running the tests)

Chunk size	Correct	Wrong	Correctly refused	Wrongly refused
300	_ / 10	_	_	_
800	_ / 10	_	_	_

What I learned: write 2-3 lines here, for example which chunk size worked better and why, and one question the system got wrong and what caused it.

Limitations
Text inside images is not read. The PDF contains graphics, and pypdf extracts only real text. Questions about content that exists only as an image can't be answered.
Tables can extract in a jumbled order.
Chunking is by character count, so a chunk can cut a sentence in half. Overlap reduces this but doesn't remove it.
Free-tier API limits: the Gemini free tier has a small daily request limit per model, so the live demo may occasionally return a rate-limit error. Answers are cached locally during testing to save quota.
Only one document is searched at a time.
Possible improvements
Hybrid search (keyword + vector) and a reranker for better retrieval.
Sentence- or section-aware chunking instead of fixed-size chunks.
OCR for scanned or image-based pages.
A larger evaluation set with automatic scoring.
Source of the document

The sample PDF is the publicly available Optima Secure brochure by HDFC Ergo. All rights belong to the original owner; it is used here for demonstration only.
