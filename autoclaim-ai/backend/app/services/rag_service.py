import hashlib
import json
import math
from pathlib import Path
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.models.schemas import PolicyCitation

EMBED_DIM = 768


class LocalRAGService:
    """Qdrant vector store + local Ollama embeddings (with deterministic fallback)."""

    def __init__(self, in_memory: bool | None = None) -> None:
        use_mem = settings.qdrant_in_memory if in_memory is None else in_memory
        self.client = QdrantClient(":memory:") if use_mem else QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def embed_text(self, text: str) -> list[float]:
        """Call local Ollama embedding model; fall back to deterministic hash vector if offline."""
        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.ollama_embed_model, "prompt": text},
                timeout=10.0,
            )
            if resp.status_code == 200:
                emb = resp.json().get("embedding")
                if isinstance(emb, list) and len(emb) == EMBED_DIM:
                    return emb
        except Exception:
            pass
        return self._fallback_embedding(text)

    @staticmethod
    def _fallback_embedding(text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        tokens = [t.strip(".,:;!?()[]{}") for t in text.lower().split() if t.strip()]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % EMBED_DIM
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def ingest_documents(self, docs: list[dict]) -> int:
        points: list[PointStruct] = []
        for doc in docs:
            doc_id = doc.get("doc_id") or str(uuid.uuid4())
            full_text = f"{doc['title']} - {doc['section']}: {doc['content']}"
            vector = self.embed_text(full_text)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "doc_id": doc_id,
                        "title": doc["title"],
                        "section": doc["section"],
                        "content": doc["content"],
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def seed_default_policies(self) -> int:
        info = self.client.get_collection(self.collection)
        if (info.points_count or 0) > 0:
            return 0
        data_path = Path(__file__).resolve().parents[2] / "data" / "policies.json"
        if not data_path.exists():
            return 0
        docs = json.loads(data_path.read_text(encoding="utf-8"))
        return self.ingest_documents(docs)

    def search(self, query: str, limit: int = 3) -> list[PolicyCitation]:
        query_vec = self.embed_text(query)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=limit,
        ).points
        citations: list[PolicyCitation] = []
        for hit in hits:
            payload = hit.payload or {}
            citations.append(
                PolicyCitation(
                    doc_id=str(payload.get("doc_id", "UNKNOWN")),
                    title=str(payload.get("title", "Policy Document")),
                    section=str(payload.get("section", "General")),
                    excerpt=str(payload.get("content", "")),
                    score=round(float(hit.score), 3),
                )
            )
        return citations
