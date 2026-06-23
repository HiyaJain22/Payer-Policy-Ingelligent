"""
retrieval.py — Hybrid dense/BM25 retrieval with RRF fusion and MMR diversification
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Per-parameter seed queries
# ──────────────────────────────────────────────────────────────────────────────

PARAM_QUERIES: Dict[str, str] = {
    "Age": "age eligibility years of age older pediatric adult",
    "Step Therapy Requirements Documented in Policy":
        "step therapy inadequate response intolerance contraindication trial fail first biologic prior treatment",
    "Number of Steps through Brands":
        "number of biologic steps prior biologic targeted synthetic",
    "Number of Steps through Generic":
        "generic steps methotrexate cyclosporine acitretin conventional",
    "Step through-Phototherapy": "phototherapy uvb puva requirement step through",
    "TB Test required": "tb test tuberculosis igra tst quantiferon",
    "Quantity Limits": "quantity limit vial syringe dose limit per days",
    "Specialist Types": "dermatologist rheumatologist prescribed by specialist",
    "Initial Authorization Duration(in-months)": "initial authorization duration months approval",
    "Reauthorization Duration(in-months)": "renewal reauthorization duration months",
    "Reauthorization Required": "reauthorization renewal required continuation",
    "Reauthorization Requirements Documented in Policy":
        "clinical response improvement BSA symptoms baseline continuation criteria",
}


# ──────────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        embeddings: np.ndarray,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_top_k: int = 30,
        bm25_top_k: int = 30,
        rrf_k: int = 60,
        mmr_top_k: int = 3,
        mmr_lambda: float = 0.7,
        global_top_chunks: int = 5,
    ):
        self.metadata = metadata
        self.embeddings = embeddings
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.mmr_top_k = mmr_top_k
        self.mmr_lambda = mmr_lambda
        self.global_top_chunks = global_top_chunks

        # Normalise metadata fields
        for m in self.metadata:
            m["content"] = re.sub(r"\s+", " ", str(m.get("content", ""))).strip()
            m["pdf_name"] = re.sub(r"\s+", " ", str(m.get("pdf_name", ""))).strip()
            m["brand_name_list"] = (
                [b.upper() for b in m["brand_name"]]
                if isinstance(m.get("brand_name"), list)
                else [str(m.get("brand_name", "")).upper()]
            )

        self.model = SentenceTransformer(embedding_model_name)

    # ── Candidate Filtering ───────────────────────────────────────────────────

    def get_candidates(self, filename: str) -> List[int]:
        filename = filename.strip()
        return [i for i, m in enumerate(self.metadata) if m["pdf_name"] == filename]

    # ── Dense Search ─────────────────────────────────────────────────────────

    def dense_search(self, query: str, candidate_ids: List[int]) -> List[Tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sub = self.embeddings[candidate_ids]
        scores = np.dot(sub, q)
        idx = np.argsort(scores)[::-1][: self.dense_top_k]
        return [(candidate_ids[i], float(scores[i])) for i in idx]

    # ── BM25 Search ───────────────────────────────────────────────────────────

    def bm25_search(self, query: str, candidate_ids: List[int]) -> List[Tuple[int, float]]:
        docs = [self.metadata[i]["content"] for i in candidate_ids]
        tok = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tok)
        scores = bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][: self.bm25_top_k]
        return [(candidate_ids[i], float(scores[i])) for i in idx]

    # ── RRF Fusion ────────────────────────────────────────────────────────────

    def rrf(self, lists: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for lst in lists:
            for rank, (doc, _) in enumerate(lst, start=1):
                scores[doc] = scores.get(doc, 0) + 1 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── MMR Diversification ───────────────────────────────────────────────────

    def mmr(self, query: str, doc_ids: List[int]) -> List[int]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        doc_embs = self.embeddings[doc_ids]
        selected, remaining = [], list(range(len(doc_ids)))
        rel = np.dot(doc_embs, q_emb)
        first = int(np.argmax(rel))
        selected.append(first)
        remaining.remove(first)
        while remaining and len(selected) < self.mmr_top_k:
            best, best_score = None, -1.0
            for i in remaining:
                relevance = float(np.dot(doc_embs[i], q_emb))
                diversity = max(float(np.dot(doc_embs[i], doc_embs[j])) for j in selected)
                score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * diversity
                if score > best_score:
                    best_score, best = score, i
            selected.append(best)
            remaining.remove(best)
        return [doc_ids[i] for i in selected]

    # ── Main Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self, filename: str, brand: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        candidates = self.get_candidates(filename)
        final_ids: set = set()
        param_to_chunk_ids: Dict[str, List[int]] = {}

        for param, query in PARAM_QUERIES.items():
            q = f"{brand} {query}"
            d = self.dense_search(q, candidates)
            b = self.bm25_search(q, candidates)
            fused = self.rrf([d, b])
            top_ids = [doc for doc, _ in fused[:15]]
            mmr_ids = self.mmr(q, top_ids)
            param_to_chunk_ids[param] = mmr_ids
            final_ids.update(mmr_ids)

        # Global rerank
        q_global = f"{brand} plaque psoriasis policy criteria"
        q_emb = self.model.encode([q_global], normalize_embeddings=True)[0]
        scored = sorted(
            [(i, float(np.dot(self.embeddings[i], q_emb))) for i in final_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        final_chunks = [self.metadata[i] for i, _ in scored[: self.global_top_chunks]]
        return final_chunks, param_to_chunk_ids
