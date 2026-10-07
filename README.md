📄 Ask My PDF: a RAG Q&A app over an insurance policy

A small question-answering app that reads a PDF, finds the parts relevant to your question, and answers using only those parts, with the page numbers it used. If the document doesn't contain the answer, it says so instead of guessing.

I built it to understand how LLMs and Retrieval-Augmented Generation (RAG) actually work end to end. The sample document is a public health insurance brochure (HDFC Ergo Optima Secure), but the pipeline works with any text-based PDF.

The idea in plain words

An LLM like ChatGPT or Gemini doesn't know what's inside your specific document, and if you ask anyway it may just guess. RAG fixes this in two steps:

Retrieve: search the document for the passages most relevant to the question.
Generate: give only those passages to the LLM and tell it to answer from them.

That keeps answers grounded in the source, makes them checkable (page numbers), and lets the model admit when it can't find something.

How it works
PDF
 └─► read text page by page (pypdf), clean up ligatures like "ﬁ"
      └─► split into overlapping chunks (default 500 characters, 50 overlap), keeping page numbers
           └─► turn each chunk into an embedding (sentence-transformers, all-MiniLM-L6-v2)
                └─► store the vectors in ChromaDB

Question
 └─► embed the question
      └─► find the top-k most similar chunks in ChromaDB
           └─► build a prompt: "answer ONLY from this context, cite pages, otherwise say you couldn't find it"
                └─► LLM writes the answer
                     └─► show the answer plus the sources (page, similarity score)
Tech stack
Part	Tool
PDF reading	pypdf
Embeddings	sentence-transformers (all-MiniLM-L6-v2, runs locally)
Vector store	ChromaDB (in-memory, cosine similarity)
LLM	Google Gemini API (tested). OpenRouter and Ollama are also supported in the code but I haven't tested them.
UI	Streamlit
Project files
├── rag_core.py          # chunking, indexing, retrieval, LLM call (retries + caching)
├── app.py               # Streamlit web UI (upload any PDF and ask questions)
├── run_tests.py         # runs a list of test questions and saves the answers to a CSV
├── check_pdf.py         # prints a few chunks to check the PDF text extracted properly
├── requirements.txt
└── policy.pdf           # sample document (public brochure)
Run it yourself

1. Install

bash
git clone https://github.com/VasanthiModepalli/insurance-policy-rag.git
cd insurance-policy-rag
pip install -r requirements.txt

2. Get a free Gemini API key from Google AI Studio and set it as an environment variable (the key is never stored in the code):

bash
# Windows (cmd)
set GEMINI_API_KEY=your_key_here
set LLM_MODEL=gemini-3.8-flash

# Mac / Linux
export GEMINI_API_KEY=your_key_here
export LLM_MODEL=gemini-3.8-flash

Model names change often, so if you get a "model not found" error, pick a current model name from Google AI Studio and set LLM_MODEL to it.

3. Start the app

bash
streamlit run app.py

Upload any text-based PDF (or use the built-in policy.pdf), then ask questions. The sidebar lets you change the chunk size and how many chunks are retrieved (top-k).

Other options

bash
python check_pdf.py          # check the PDF extracts into readable chunks
python debug_retrieval.py    # see which chunks are retrieved for a question (free, no LLM)
python run_tests.py          # run the test questions and save a CSV
What I observed while testing

Answers to questions that are in the document came back correct and cited pages:

Question	Answer (abridged)
Is there an extra charge for the restored amount?	No extra charge; the restored cover is available for later claims at no additional cost (page 5).
How many times can cover be restored in a year?	Once per policy year (page 5). The model also noticed an "Unlimited Restore" add-on on page 15, which I hadn't expected.

One failure is worth noting. For the vague question "What do the benefits together give?", the brochure does say "4X coverage", but the app replied "I couldn't find this in the document." The model behaved correctly given what it was shown. The problem was retrieval: the question shares few words with the sentence that holds the answer, so the right chunk wasn't among the top results. debug_retrieval.py lets you inspect exactly what was retrieved.

I did not run a full scored evaluation, so I'm not reporting accuracy percentages.

What I learned
An LLM only knows what's in its prompt. RAG is mostly about choosing what goes into that prompt.
Most failures are retrieval failures, not model failures. Check what was retrieved before blaming the LLM.
Chunk size and overlap change what the model can see. Too small and the answer is split up; too large and the chunk is noisy.
The prompt controls hallucination. Telling the model to answer only from the context and to admit when it can't is what produces an honest "I couldn't find this."
Real-world details matter: PDFs contain odd characters (the "ﬁ" ligature broke words until I normalised the text), free API tiers have rate limits and daily quotas, and servers return 503s. I added retries, local answer caching, and a way to switch model or provider using environment variables.
Test with questions that have no answer in the document. That's how you check whether the system makes things up.
Using it on other documents

Nothing in the pipeline is specific to insurance. To use it on something else, change the document (upload another PDF in the app) and, if you want, the instructions in the prompt in rag_core.py. The same structure works for HR policies, product manuals, contracts, research papers or study notes.

Limitations
Text inside images isn't read. pypdf extracts real text only, so scanned pages or text embedded in graphics are invisible. OCR would be needed.
Tables can come out in a jumbled order.
Chunks are cut by character count, so a chunk can end mid-sentence.
One document at a time.
English-focused embeddings. A multilingual embedding model would be needed for other languages.
Free API limits: the Gemini free tier allows only a small number of requests per day per model, so heavy use can hit rate-limit errors.
Ideas for next steps
Hybrid search (keyword + vector) and a reranker, to fix misses like the "4X coverage" question.
Support several documents at once, with file names in the citations.
Sentence- or section-aware chunking.
A larger, scored test set.
OCR for scanned PDFs.
About the document

The sample PDF is the publicly available Optima Secure brochure by HDFC Ergo. All rights belong to the original owner; it is included for demonstration only.
