"""
Run a fixed set of test questions at different chunk sizes and save results.csv.
Open results.csv in Excel and fill the 'verdict' column by hand:
  correct / wrong / correctly refused / wrongly refused
Then copy the summary into your README.

Run:  python run_tests.py
"""

import csv
import time

from rag_core import answer, build_index, load_chunks

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

with open("results.csv", "w", newline="", encoding="utf-8") as f:
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
            time.sleep(2)  # stay within free-tier rate limits

print("\nDone. Open results.csv and fill the verdict column.")
