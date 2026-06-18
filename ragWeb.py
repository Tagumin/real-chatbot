"""
rag.py
======
Core RAG Engine for Indonesian Legal Assistant

Improvements:
- Safe imports (LangChain Ollama compatible)
- Stable output parsing
- Safer retriever handling
- Better context building
- Robust cleaning & citation
"""

import re
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vector import retriever_law, retriever_culture


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
LLM_MODEL = "qwen2.5:3b"
TEMPERATURE = 0.1
MAX_CONTEXT_CHARS = 8000

MODE_LAW = "1"
MODE_CULTURE = "2"


# ─────────────────────────────
# LLM INIT
# ─────────────────────────────
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=TEMPERATURE
)


# ─────────────────────────────
# PROMPT
# ─────────────────────────────
PROMPT = """
You are an Indonesian legal assistant.

Use ONLY the provided context to answer.
If no relevant information is found, respond:
"No relevant legal provision found."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(PROMPT)

chain = prompt | llm | StrOutputParser()


# ─────────────────────────────
# RETRIEVER SELECTOR
# ─────────────────────────────
def get_retriever(mode: str):
    if mode == MODE_LAW:
        return retriever_law, "LAW"
    elif mode == MODE_CULTURE:
        return retriever_culture, "CULTURE"
    return None, None


# ─────────────────────────────
# BUILD CONTEXT
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

    context = "\n\n---\n\n".join(parts)

    return context[:MAX_CONTEXT_CHARS]


# ─────────────────────────────
# CLEAN OUTPUT
# ─────────────────────────────
def clean(text: str):
    if not text:
        return ""

    # remove hidden reasoning tags if any
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


# ─────────────────────────────
# POST PROCESS
# ─────────────────────────────
def post_process(answer: str):
    if not answer:
        return "No relevant legal provision found."

    low = answer.lower()

    if "no relevant legal provision found" in low:
        return answer

    if len(answer.strip()) < 15:
        return "No relevant legal provision found."

    return answer


# ─────────────────────────────
# CITATION HANDLER
# ─────────────────────────────
def add_citation(answer: str, docs):
    if not docs:
        return answer

    # If already has article reference, keep it
    if re.search(r'Article\s+\d+', answer):
        return answer

    first = docs[0].metadata.get("article", "")

    if not first or first == "-":
        match = re.search(r'Article\s+(\d+[A-Za-z]*)', docs[0].page_content)
        first = f"Article {match.group(1)}" if match else "Unknown"

    return f"{answer}\n\nSource: {first}"


# ─────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────
def ask_question(question: str, mode: str = MODE_LAW):
    start = datetime.now()

    retriever, domain = get_retriever(mode)

    if retriever is None:
        return {
            "answer": "Invalid mode selected",
            "domain": None,
            "docs": 0,
            "time": 0
        }

    # safer retrieval (supports multiple retriever types)
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "domain": domain,
            "docs": 0,
            "time": (datetime.now() - start).total_seconds()
        }

    context = build_context(docs)

    raw_answer = chain.invoke({
        "context": context,
        "question": question
    })

    answer = clean(raw_answer)
    answer = post_process(answer)
    answer = add_citation(answer, docs)

    return {
        "answer": answer,
        "domain": domain,
        "docs": len(docs),
        "time": (datetime.now() - start).total_seconds()
    }