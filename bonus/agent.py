"""HybridMemoryAgent — Kết hợp Vector Store (episodic memory) và Feature Store (user profile).

Minimal POC cho Lab 19 Bonus Challenge.
Sử dụng: fastembed + Qdrant in-memory + BM25 + Feast (SQLite online store).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastembed import TextEmbedding
from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION = "user_memories"
RRF_K = 60

# Feast repo path — relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEAST_REPO = _PROJECT_ROOT / "app" / "feast_repo"

# Features to retrieve from Feast online store
PROFILE_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
]
VELOCITY_FEATURES = [
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


def _sentence_split(text: str) -> list[str]:
    """Chunk text vào các câu — sentence-level chunking strategy.

    Dùng regex đơn giản tách theo dấu chấm / chấm hỏi / chấm than.
    Trong production nên dùng underthesea hoặc spaCy sentence splitter.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Gộp câu quá ngắn (< 20 ký tự) với câu trước
    merged: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if merged and len(merged[-1]) < 20:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged if merged else [text.strip()]


class HybridMemoryAgent:
    """Agent kết hợp episodic memory (Vector Store) và stable profile (Feature Store).

    - remember(text, user_id): Chunk text → embed → upsert vào Qdrant.
    - recall(query, user_id): Hybrid search (BM25 + vector + RRF) trên Qdrant
      + Feast online lookup → assembled context string.
    """

    def __init__(self) -> None:
        # Embedding model
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)

        # Qdrant in-memory
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

        # BM25 index — rebuilt on each remember() call
        self._all_chunks: list[dict] = []  # {"id": int, "text": str, "user_id": str}
        self._bm25: BM25Okapi | None = None
        self._next_id = 0

        # Feast feature store
        self.fs = FeatureStore(repo_path=str(FEAST_REPO))

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user.

        Chunks text theo câu → embed → upsert vào Qdrant collection
        với payload user_id để hỗ trợ per-user filtering.
        """
        chunks = _sentence_split(text)
        vectors = list(self.embedder.embed(chunks))

        points: list[PointStruct] = []
        for chunk_text, vec in zip(chunks, vectors):
            point_id = self._next_id
            self._next_id += 1
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={"user_id": user_id, "text": chunk_text},
                )
            )
            self._all_chunks.append(
                {"id": point_id, "text": chunk_text, "user_id": user_id}
            )

        self.client.upsert(collection_name=COLLECTION, points=points)

        # Rebuild BM25 index (toàn bộ chunks hiện có)
        tokenized = [c["text"].lower().split() for c in self._all_chunks]
        self._bm25 = BM25Okapi(tokenized)

        print(f"  [remember] Stored {len(chunks)} chunk(s) for user={user_id}")

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features → return assembled context.

        1. Get user profile + recent activity from Feast online store
        2. Hybrid search Qdrant filtered by user_id (BM25 + vector + RRF)
        3. Assemble context string
        """
        # ── Step 1: Feast online lookup ──────────────────────────────────
        try:
            features = self.fs.get_online_features(
                features=PROFILE_FEATURES + VELOCITY_FEATURES,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            profile = {k: v[0] for k, v in features.items()}
        except Exception:
            profile = {
                "user_id": user_id,
                "reading_speed_wpm": None,
                "preferred_language": None,
                "topic_affinity": None,
                "queries_last_hour": None,
                "distinct_topics_24h": None,
            }

        # ── Step 2: Hybrid search on episodic memory ─────────────────────
        top_memories = self._hybrid_search(query, user_id, top_k=3)

        # ── Step 3: Assemble context string ──────────────────────────────
        profile_str = (
            f"User profile: "
            f"language={profile.get('preferred_language', 'N/A')}, "
            f"reading_speed={profile.get('reading_speed_wpm', 'N/A')}wpm, "
            f"topic_affinity={profile.get('topic_affinity', 'N/A')}"
        )
        activity_str = (
            f"Recent activity: "
            f"queries_last_hour={profile.get('queries_last_hour', 'N/A')}, "
            f"distinct_topics_24h={profile.get('distinct_topics_24h', 'N/A')}"
        )
        if top_memories:
            memories_str = "Top memories:\n" + "\n".join(
                f"  {i+1}. {m}" for i, m in enumerate(top_memories)
            )
        else:
            memories_str = "Top memories: (no memories found)"

        context = f"{profile_str}\n{activity_str}\n{memories_str}"
        return context

    def _hybrid_search(
        self, query: str, user_id: str, top_k: int = 3
    ) -> list[str]:
        """Hybrid search: BM25 + Vector + RRF, filtered by user_id."""
        if not self._all_chunks or self._bm25 is None:
            return []

        depth = max(top_k * 5, 20)

        # ── BM25 keyword search ──────────────────────────────────────────
        user_indices = [
            i for i, c in enumerate(self._all_chunks) if c["user_id"] == user_id
        ]
        if not user_indices:
            return []

        bm25_scores = self._bm25.get_scores(query.lower().split())
        # Filter and rank by user
        user_scored = [(idx, bm25_scores[idx]) for idx in user_indices]
        user_scored.sort(key=lambda x: -x[1])
        kw_ids = [self._all_chunks[idx]["text"] for idx, _ in user_scored[:depth]]

        # ── Vector semantic search ───────────────────────────────────────
        q_vec = next(self.embedder.embed([query])).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=depth,
        )
        sem_ids = [p.payload["text"] for p in results.points]

        # ── RRF fusion ───────────────────────────────────────────────────
        rrf: dict[str, float] = {}
        for rank, text in enumerate(kw_ids, start=1):
            rrf[text] = rrf.get(text, 0.0) + 1.0 / (RRF_K + rank)
        for rank, text in enumerate(sem_ids, start=1):
            rrf[text] = rrf.get(text, 0.0) + 1.0 / (RRF_K + rank)

        sorted_results = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [text for text, _ in sorted_results]
