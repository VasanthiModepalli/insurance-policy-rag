"""
Run a fixed set of test questions at different chunk sizes and save a results CSV.
Open the CSV in Excel and fill the 'verdict' column by hand:
  correct / wrong / correctly refused / wrongly refused
Then copy the summary into your README.

Run:  python run_tests.py
"""

import csv
import os
import time
from datetime import datetime

from rag_core import LLM_MODEL, answer, build_index, load_chunks

PDF_PATH = "policy.pdf"   # <- put your PDF in this folder with this name
CHUNK_SIZES = [300, 800]  # the two settings you are comparing

# (question, expected answer). Use "NOT IN DOCUMENT" for refusal tests.
# Verify with Ctrl+F in the PDF that those answers are really absent.
TESTS = [
    ("What is the Plus Benefit?",
     "Base cover increases by 50% after 1 year and 100% after 2 years"),
    ("Does the Plus Benefit depend on not making claims?",
     "No, applies irrespective of claims"),
    ("When does the Restore Benefit apply?",
     "When any claim, partial or total, is made during the policy year"),
    ("How much of the base cover is restored?", "100%"),
    ("How many times can cover be restored in a year?",
     "Once per policy year"),
    ("Is there an extra charge for the restored amount?",
     "No, at no additional cost"),
    ("What do the benefits together give?", "4X coverage"),
    ("If my base cover is 10 lakh, what is it after 2 years with Plus Benefit?",
     "20 lakh"),
    ("Does the policy cover cosmetic surgery abroad?", "NOT IN DOCUMENT"),
    ("What is the capital of France?", "NOT IN DOCUMENT"),
]

# A NEW file every run (model + time in the name), so old results are never overwritten
SAFE_MODEL = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in LLM_MODEL)
OUT_FILE = f"results_{SAFE_MODEL}_{datetime.now():%Y%m%d_%H%M}.csv"

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["chunk_size", "question", "expected", "answer", "pages", "verdict"])

    for size in CHUNK_SIZES:
        chunks = load_chunks(PDF_PATH, chunk_size=size, overlap=50)
        collection = build_index(chunks)
        print(f"\n##### chunk_size={size} ({len(chunks)} chunks) #####")

        for question, expected in TESTS:
            text, sources = answer(question, collection)
            pages = sorted({s["page"] for s in sources})
            writer.writerow([size, question, expected, text, pages, ""])
            f.flush()  # save progress so a crash doesn't lose results
            print(f"\nQ: {question}\nExpected: {expected}\nGot: {text}")
            # Pause between questions to respect free-tier per-minute limits.
            # Slow it down if you still get errors:  set TEST_DELAY=15
            time.sleep(float(os.environ.get("TEST_DELAY", "2")))

print(f"\nDone. Open {OUT_FILE} and fill the verdict column.")
