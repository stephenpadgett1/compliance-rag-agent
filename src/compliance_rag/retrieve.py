"""Hybrid retrieval — vector search + explicit-ID keyword lookup.

Why hybrid:
- Vector search handles "what evidence would an auditor want for account management?"
  (the query never says "AC-2" but should retrieve it).
- ID lookup handles "what's the difference between AC-2 and AC-3?" — when the user
  names criteria, we should not rely on the embedding model to find them.

Strategy:
1. Embed the query and pull top-K via vector similarity.
2. Regex-scan the query for criterion IDs; if found, fetch each by ID, plus its parent
   (if it's an enhancement) and any enhancements (if it's a root).
3. Merge, dedupe by ID, keep top-K total.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from compliance_rag.config import Config
from compliance_rag.corpora.base import CorpusChunk
from compliance_rag.ingest import collection_name
from compliance_rag.usage import CallStats, UsageStats

# Matches NIST 800-53 (AC-2, AC-2(1)) and AICPA TSC (CC6.1, A1.2, PI1.1) styles.
# Enhancement form goes first — the regex engine returns the first matching
# alternative, so without this ordering "AC-6(1)" would match as just "AC-6".
CRITERION_ID_PATTERN = re.compile(
    r"\b[A-Z]{2}-\d+\(\d+\)"  # NIST enhancement: AC-6(1)
    r"|\b[A-Z]{2}-\d+\b"  # NIST root: AC-2
    r"|\b[A-Z]{1,3}\d+\.\d+\b"  # AICPA TSC: CC6.1, A1.2, PI1.1
)


@dataclass(frozen=True)
class RetrievalResult:
    chunk: CorpusChunk
    score: float | None  # None for ID-direct hits; lower is closer for vector hits
    source: str  # "vector" or "id-direct"


class Retriever:
    """Stateful retriever — holds the Chroma collection and embedding client."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._openai = OpenAI(api_key=config.openai_api_key)
        self._chroma = chromadb.PersistentClient(
            path=str(config.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_collection(name=collection_name(config.corpus))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run hybrid retrieval, returning up to `top_k` results."""
        results, _ = self.retrieve_with_stats(query, top_k=top_k)
        return results

    def retrieve_with_stats(
        self, query: str, top_k: int = 5
    ) -> tuple[list[RetrievalResult], UsageStats]:
        """Same as retrieve() but also returns the embedding-call usage stats."""
        explicit_ids = CRITERION_ID_PATTERN.findall(query)
        id_hits = self._fetch_by_ids(explicit_ids) if explicit_ids else []

        # Reserve some slots for ID hits; fill the rest with vector results
        vector_budget = max(top_k - len(id_hits), top_k // 2)
        vector_hits, stats = self._vector_search(query, top_k=vector_budget)

        return self._merge(id_hits + vector_hits, top_k=top_k), stats

    def _vector_search(self, query: str, top_k: int) -> tuple[list[RetrievalResult], UsageStats]:
        embedding_response = self._openai.embeddings.create(
            model=self._config.embedding_model, input=[query]
        )
        embedding = embedding_response.data[0].embedding
        stats = UsageStats()
        stats.record(
            CallStats.from_openai_embedding(
                model=self._config.embedding_model,
                total_tokens=embedding_response.usage.total_tokens,
            )
        )

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            RetrievalResult(
                chunk=_chunk_from_chroma(doc, meta),
                score=dist,
                source="vector",
            )
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
        return hits, stats

    def _fetch_by_ids(self, ids: list[str]) -> list[RetrievalResult]:
        # Dedupe input while preserving order
        seen: set[str] = set()
        unique_ids = [i for i in ids if not (i in seen or seen.add(i))]
        if not unique_ids:
            return []

        # Expand: for any explicit ID, also pull its parent (if enhancement) or children
        # (if root). We do this with a second `get()` against `parent_id` metadata.
        result = self._collection.get(ids=unique_ids, include=["documents", "metadatas"])
        primaries = [
            RetrievalResult(chunk=_chunk_from_chroma(doc, meta), score=None, source="id-direct")
            for doc, meta in zip(result["documents"], result["metadatas"], strict=True)
        ]

        # Pull parents of any enhancement hits (one extra fetch is fine).
        parent_ids: set[str] = set()
        for r in primaries:
            if r.chunk.parent_id and r.chunk.parent_id not in unique_ids:
                parent_ids.add(r.chunk.parent_id)
        parents: list[RetrievalResult] = []
        if parent_ids:
            parent_result = self._collection.get(
                ids=list(parent_ids), include=["documents", "metadatas"]
            )
            parents = [
                RetrievalResult(
                    chunk=_chunk_from_chroma(doc, meta), score=None, source="id-direct"
                )
                for doc, meta in zip(
                    parent_result["documents"], parent_result["metadatas"], strict=True
                )
            ]
        return primaries + parents

    @staticmethod
    def _merge(results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Dedupe by criterion_id, prioritize ID-direct hits over vector."""
        # Sort: id-direct first, then by ascending vector distance (closer = better)
        ranked = sorted(
            results,
            key=lambda r: (r.source != "id-direct", r.score if r.score is not None else 0),
        )
        seen: set[str] = set()
        out: list[RetrievalResult] = []
        for r in ranked:
            if r.chunk.criterion_id in seen:
                continue
            seen.add(r.chunk.criterion_id)
            out.append(r)
            if len(out) >= top_k:
                break
        return out


def _chunk_from_chroma(document: str, metadata: dict) -> CorpusChunk:
    """Reconstruct a CorpusChunk from Chroma's stored row."""
    return CorpusChunk(
        criterion_id=metadata["criterion_id"],
        title=metadata.get("title", ""),
        text=document,
        parent_id=metadata.get("parent_id"),
        page_number=metadata.get("page_number"),
        extra={
            k: v
            for k, v in metadata.items()
            if k not in {"criterion_id", "title", "parent_id", "page_number", "is_root"}
        },
    )
