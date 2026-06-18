

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import numpy as np

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
EMBED_MODEL = "bge-m3"

DBS = {
    "law": {
        "path": "./vectorstore",
        "collection": "indonesian_law_rag"
    },
    "culture": {
        "path": "./vectorstore_culture",
        "collection": "balinese_culture_rag"
    }
}

VECTOR_TOP_K = 10
BM25_TOP_K   = 10
RERANK_TOP_K = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────
# EMBEDDINGS
# ─────────────────────────────
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# ─────────────────────────────
# VECTOR STORES
# ─────────────────────────────
vectorstores: Dict[str, Chroma] = {}

for name, cfg in DBS.items():
    vectorstores[name] = Chroma(
        persist_directory=cfg["path"],
        collection_name=cfg["collection"],
        embedding_function=embeddings
    )

# ─────────────────────────────
# BM25 RETRIEVER
# ─────────────────────────────
class BM25Retriever:
    def __init__(self, vectorstore: Chroma, domain: str):
        self.domain = domain
        print(f"📊 Loading BM25 index: {domain}")

        data = vectorstore.get()

        self.docs: List[Document] = []
        for i in range(len(data["ids"])):
            self.docs.append(
                Document(
                    page_content=data["documents"][i],
                    metadata=data["metadatas"][i] or {}
                )
            )

        tokenized = [d.page_content.lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

        print(f"✅ BM25 ready: {domain} ({len(self.docs)} docs)")

    def invoke(self, query: str) -> List[Document]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:BM25_TOP_K]

        results = []
        for i in top_idx:
            if scores[i] > 0:
                doc = self.docs[i]
                doc.metadata["bm25_score"] = float(scores[i])
                doc.metadata["domain"] = self.domain
                results.append(doc)

        return results


# ─────────────────────────────
# CROSS ENCODER RERANKER
# ─────────────────────────────
class Reranker:
    def __init__(self):
        print("🎯 Loading reranker...")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("✅ Reranker ready")

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)

        order = np.argsort(scores)[::-1][:top_k]

        result = []
        for rank, idx in enumerate(order):
            doc = docs[idx]
            doc.metadata["rerank_score"] = float(scores[idx])
            doc.metadata["rerank_rank"] = rank + 1
            result.append(doc)

        return result


# ─────────────────────────────
# DOMAIN RETRIEVER (INDIVIDUAL)
# ─────────────────────────────
class DomainRetriever:
    def __init__(self, vectorstore, bm25, reranker, domain: str):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.reranker = reranker
        self.domain = domain

    def invoke(self, query: str) -> List[Document]:

        vec_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": VECTOR_TOP_K}
        ).invoke(query)

        bm25_docs = self.bm25.invoke(query)

        all_docs = vec_docs + bm25_docs

        # dedup
        seen = {}
        for d in all_docs:
            key = d.page_content[:120]
            d.metadata["domain"] = self.domain
            seen[key] = d

        return self.reranker.rerank(query, list(seen.values()), RERANK_TOP_K)


# ─────────────────────────────
# HYBRID MULTI DOMAIN RETRIEVER
# ─────────────────────────────
class HybridRetriever:
    def __init__(self):
        print("\n🔧 Init Hybrid Retriever (MULTI DOMAIN)")

        self.vectorstores = vectorstores
        self.reranker = Reranker()

        self.bm25 = {
            name: BM25Retriever(vs, name)
            for name, vs in vectorstores.items()
        }

        print("✅ System ready (law + culture)\n")

    def invoke(self, query: str) -> List[Document]:

        all_candidates = {}

        for domain, vs in self.vectorstores.items():

            vec_docs = vs.as_retriever(
                search_kwargs={"k": VECTOR_TOP_K}
            ).invoke(query)

            bm25_docs = self.bm25[domain].invoke(query)

            for d in vec_docs + bm25_docs:
                key = (d.metadata.get("chunk_id"), d.page_content[:120])
                d.metadata["domain"] = domain
                all_candidates[key] = d

        merged = list(all_candidates.values())
        return self.reranker.rerank(query, merged, RERANK_TOP_K)

    # ─────────────────────────────
    # INI YANG KAMU CARI
    # ─────────────────────────────
    def get_domain_retriever(self, domain: str):
        if domain not in self.vectorstores:
            raise ValueError(f"Domain tidak ditemukan: {domain}")

        return DomainRetriever(
            vectorstore=self.vectorstores[domain],
            bm25=self.bm25[domain],
            reranker=self.reranker,
            domain=domain
        )


# ─────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────
retriever = HybridRetriever()

# ACCESS LANGSUNG (INI YANG KAMU MAU)
retriever_law = retriever.get_domain_retriever("law")
retriever_culture = retriever.get_domain_retriever("culture")