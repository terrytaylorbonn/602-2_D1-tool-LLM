# ai_demo_05_rag.py
# Basic RAG demo:
# retrieve relevant text
# inject retrieved text into LLM prompt
# LLM answers from retrieved context

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------
# Load API key
# -----------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------
# Demo documents
# -----------------------------------

DOCS_DIR = Path("rag_docs")
DOCS_DIR.mkdir(exist_ok=True)

(DOCS_DIR / "taipei_shipments.txt").write_text(
    "Truck 12 delayed in Taipei due to flooding. "
    "Truck 18 is on schedule in Taipei.",
    encoding="utf-8"
)

(DOCS_DIR / "supplier_notes.txt").write_text(
    "Supplier A reported an outage affecting brake components.",
    encoding="utf-8"
)

(DOCS_DIR / "weather_notes.txt").write_text(
    "Heavy rain caused flooding near Taipei logistics routes.",
    encoding="utf-8"
)

# -----------------------------------
# Simple retrieval tool
# -----------------------------------

def retrieve_docs(query, top_k=2):
    query_words = set(query.lower().split())
    scored_docs = []

    for path in DOCS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        text_words = set(text.lower().split())

        score = len(query_words.intersection(text_words))

        scored_docs.append({
            "filename": path.name,
            "score": score,
            "text": text
        })

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]

# -----------------------------------
# User question
# -----------------------------------

user_question = "Why is Truck 12 delayed?"

# -----------------------------------
# RAG step 1: retrieve docs
# -----------------------------------

retrieved = retrieve_docs(user_question)

context = "\n\n".join(
    f"[{doc['filename']}]\n{doc['text']}"
    for doc in retrieved
)

print("\nRETRIEVED CONTEXT:")
print(context)

# -----------------------------------
# RAG step 2: inject context into prompt
# -----------------------------------

system_prompt = """
You are an AI assistant.

Answer the user question using ONLY the retrieved context.
If the answer is not in the context, say: "I do not know from the provided context."
"""

user_prompt = f"""
Retrieved context:

{context}

User question:
{user_question}
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0
)

answer = response.choices[0].message.content

print("\nLLM ANSWER:")
print(answer)

