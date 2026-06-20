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
MAX_CONTEXT_CHARS = 8000

DEBUG = True


# ─────────────────────────────
# LLM
# ─────────────────────────────
llm = OllamaLLM(model=LLM_MODEL, temperature=TEMPERATURE)

PROMPT = """
You are a professional Indonesian Legal Assistant. Your answers must be derived entirely from your internal knowledge base. Do not invent, assume, or extrapolate any legal facts, laws, or consequences outside of what is explicitly given to you.

Follow these strict operational rules:

1. CLIENT EXPERIENCE (NO META-TALK): Never use words like "context," "dataset," "source," "provided text," or "provided information" in your response. Act as if the knowledge is entirely your own.

2. STRICT DATA ADHERENCE: Answer the user's question ONLY if the exact information required is present in your knowledge base. If a law, rule, or action is not explicitly covered, treat it as unknown. Do not use outside training knowledge to fill in gaps for legal advice.

3. OUT-OF-DATA FALLBACK: If the question cannot be fully answered using only your knowledge base, you must politely state that you do not have that specific information.
   - Good response: "I do not have the specific regulations regarding [Topic] available at this time to give you an accurate answer."
   - Bad response: "That information is not in the provided context."

4. NON-LEGAL QUESTIONS: For general, non-legal inputs (like greetings, "thank you," or casual chat), respond naturally, politely, and briefly while maintaining your professional persona.

5. FORMATTING:
   - Provide a clear, direct answer in the first sentence.
   - Use structured bullet points, short paragraphs, and bold text for key terms so the user can scan the response effortlessly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(PROMPT)
chain = prompt | llm | StrOutputParser()


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

    # Strip any stray <tool_call>...</tool_call> blocks the model might emit.
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
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


def debug_print_docs(docs, label: str):
    """Shared debug printer for a list of retrieved docs (used for both
    LAW and CULTURE results) -- replaces the two near-identical loops
    that used to exist in ask_question()."""
    if not DEBUG or not docs:
        return

    for i, d in enumerate(docs):
        print("\n" + "=" * 80)
        print(f"DOC {i + 1} ({label})")
        print("=" * 80)
        print(f"Domain       : {d.metadata.get('domain', 'N/A')}")
        print(f"Rerank Rank  : {d.metadata.get('rerank_rank', 'N/A')}")
        print(f"Rerank Score : {d.metadata.get('rerank_score', 'N/A')}")
        print(f"BM25 Score   : {d.metadata.get('bm25_score', 'N/A')}")
        print("\nContent:")
        print(d.page_content[:300])
        print("=" * 80)


def retrieve_docs(retriever, question: str):
    """Works whether the retriever exposes .invoke() (current LangChain API)
    or the older .get_relevant_documents()."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    return retriever.get_relevant_documents(question)


# ─────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────
def ask_question(question: str):
    start = datetime.now()

    try:
        debug_print("QUESTION", question)

        # ── RETRIEVAL (both domains, always) ──
        docs_law = retrieve_docs(retriever_law, question)
        docs_culture = retrieve_docs(retriever_culture, question)

        debug_print("DOCS FOUND (LAW)", len(docs_law))
        debug_print("DOCS FOUND (CULTURE)", len(docs_culture))

        debug_print_docs(docs_law, "LAW")
        debug_print_docs(docs_culture, "CULTURE")

        # ── CONTEXT ──
        context_law = build_context(docs_law)
        context_culture = build_context(docs_culture)
        debug_print("CONTEXT PREVIEW (LAW)", context_law[:500])
        debug_print("CONTEXT PREVIEW (CULTURE)", context_culture[:500])

        # ── LLM ──
        raw_law = chain.invoke({"context": context_law, "question": question})
        raw_culture = chain.invoke({"context": context_culture, "question": question})

        debug_print("RAW LLM OUTPUT (LAW)", raw_law)
        debug_print("RAW LLM OUTPUT (CULTURE)", raw_culture)

        # ── PROCESSING ──
        answer_law = clean(raw_law)
        answer_culture = clean(raw_culture)

        debug_print("CLEANED ANSWER (LAW)", answer_law)
        debug_print("CLEANED ANSWER (CULTURE)", answer_culture)

        # ── RESULT ──
        return {
            "answer_law": answer_law,
            "answer_culture": answer_culture,
            "docs_law": len(docs_law),
            "docs_culture": len(docs_culture),
            "time": (datetime.now() - start).total_seconds()
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        traceback.print_exc()

        return {
            "error": str(e),
            "answer_law": "System error occurred",
            "answer_culture": "System error occurred",
            "time": (datetime.now() - start).total_seconds()
        }


# # ─────────────────────────────
# # CLI TEST MODE
# # ─────────────────────────────
# if __name__ == "__main__":
#     print("\n🚀 RAG DEBUG MODE")
#     print("Type 'exit' to quit\n")

#     while True:
#         q = input("Ask > ")

#         if q.lower() == "exit":
#             break

#         result = ask_question(q)

#         print("\n🧠 FINAL ANSWER")
#         print("=" * 60)
#         print("LAW PERSPECTIVE:")
#         print(result.get("answer_law", result.get("error")))
#         print("\nCULTURE PERSPECTIVE:")
#         print(result.get("answer_culture", ""))
#         print("=" * 60)