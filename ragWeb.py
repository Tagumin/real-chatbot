"""
rag_debug.py
============
RAG Engine + Debug Mode for Terminal Testing

Features:
✔ Full step-by-step logging
✔ Retrieval inspection
✔ Context preview
✔ LLM raw output trace
✔ Safe error handling
"""

import re
import traceback
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vector import retriever_law, retriever_culture


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
LLM_MODEL = "qwen3:4b"
TEMPERATURE = 0.1
MAX_CONTEXT_CHARS = 14000

MODE_LAW = "1"
MODE_CULTURE = "2"

DEBUG = True


# ─────────────────────────────
# LLM
# ─────────────────────────────
llm = OllamaLLM(model=LLM_MODEL, temperature=TEMPERATURE)

PROMPT = """
You are an Indonesian legal assistant. Follow these rules in order:

1. CLASSIFY the question first:
   - LEGAL QUESTION: about Indonesian law, what is/isn't allowed, penalties, procedures, etc.
   - GENERAL QUESTION: unrelated to law (greetings, small talk, general knowledge).

2. IF the question is a LEGAL QUESTION:
   - Use ONLY the provided CONTEXT below. Do not add facts, numbers, or penalties
     that are not explicitly stated in the context.
   - If the context fully answers the question, answer it directly and cite the
     article number(s) used.
   - If the context is empty, irrelevant, or only partially answers the question,
     say so explicitly first:
     "Based on the available legal context, I could not find a provision that
     directly answers this."
     Then, ONLY if helpful, you may add ONE short sentence of general legal
     knowledge, clearly labeled as NOT from the provided law text, e.g.:
     "(General note, not from the cited law: ...)"
   - NEVER cite an article number unless its content actually supports the
     specific claim you are making. If the closest article doesn't match the
     question's scenario, treat it as "no relevant provision found" instead
     of citing it anyway.

3. IF the question is a GENERAL QUESTION (not about law):
   - Answer normally and helpfully using your own general knowledge.
   - Do not force a legal framing onto it.

4. FORMAT every legal answer using this structure, with each section on its own
   line separated by a blank line:

   **Summary**
   One short sentence answering the question directly.

   **Basis**
   The relevant article number(s) and what they actually say, in plain language.

   **Details**
   Any conditions, exceptions, or penalties, as a short bullet list if there
   are multiple points.

   Keep language clear and avoid legal jargon where a plain-language
   equivalent exists.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(PROMPT)

chain = prompt | llm | StrOutputParser()


# ─────────────────────────────
# RETRIEVER
# ─────────────────────────────
def get_retriever(mode: str):
    if mode == MODE_LAW:
        return retriever_law, "LAW"
    elif mode == MODE_CULTURE:
        return retriever_culture, "CULTURE"
    return None, None


# ─────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────
def build_context(docs):
    parts = []

    for doc in docs:
        article = doc.metadata.get("article", "")

        if not article or article == "-":
            match = re.search(r'Article\s+(\d+[A-Za-z]*)', doc.page_content)
            article = f"Article {match.group(1)}" if match else "Unknown"

        content = doc.page_content.strip()
        parts.append(f"{article}\n{content}")

    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ─────────────────────────────
# CLEAN OUTPUT
# ─────────────────────────────
def clean(text: str):
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ─────────────────────────────
# DEBUG PRINT HELPERS
# ─────────────────────────────
def debug_print(title, content):
    if not DEBUG:
        return
    print(f"\n🟡 {title}")
    print("-" * 50)
    print(content)
    print("-" * 50)


# ─────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────
def ask_question(question: str, mode: str = MODE_LAW):
    start = datetime.now()

    try:
        debug_print("QUESTION", question)
        debug_print("MODE", mode)

        retriever, domain = get_retriever(mode)

        if retriever is None:
            return {"error": "Invalid mode"}

        # ── RETRIEVAL ──
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
        else:
            docs = retriever.get_relevant_documents(question)

        debug_print("DOCS FOUND", len(docs))

        if DEBUG and docs:
            for i, d in enumerate(docs):

                rerank_score = d.metadata.get("rerank_score", "N/A")
                rerank_rank = d.metadata.get("rerank_rank", "N/A")
                bm25_score = d.metadata.get("bm25_score", "N/A")
                domain = d.metadata.get("domain", "N/A")

                print("\n" + "=" * 80)
                print(f"DOC {i+1}")
                print("=" * 80)

                print(f"Domain       : {domain}")
                print(f"Rerank Rank  : {rerank_rank}")
                print(f"Rerank Score : {rerank_score}")
                print(f"BM25 Score   : {bm25_score}")

                print("\nContent:")
                print(d.page_content[:300])

                print("=" * 80)
        if not docs:
            return {
                "answer": "No relevant documents found.",
                "domain": domain,
                "docs": 0,
                "time": (datetime.now() - start).total_seconds()
            }

        # ── CONTEXT ──
        context = build_context(docs)
        debug_print("CONTEXT PREVIEW", context[:500])

        # ── LLM ──
        raw = chain.invoke({
            "context": context,
            "question": question
        })

        debug_print("RAW LLM OUTPUT", raw)

        # ── PROCESSING ──
        answer = clean(raw)

        debug_print("CLEANED ANSWER", answer)

        # ── RESULT ──
        return {
            "answer": answer,
            "domain": domain,
            "docs": len(docs),
            "time": (datetime.now() - start).total_seconds()
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        traceback.print_exc()

        return {
            "error": str(e),
            "answer": "System error occurred",
            "time": (datetime.now() - start).total_seconds()
        }
