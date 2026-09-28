"""Vector storage with two interchangeable backends.

ChromaStore (preferred) uses chromadb PersistentClient: durable, metadata
filtering, the library interviewers expect on a RAG resume line.
NumpyStore is the zero-dependency fallback: cosine similarity with a
normalized dot product and top-k argpartition, persisted as JSONL + .npy.
Both expose the same add/query contract, so swapping backends changes nothing
upstream.
"""
import json
from pathlib import Path

import numpy as np

from .config import COLLECTION_NAME
from .loaders import Segment


def get_store(db_dir: Path):
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return NumpyStore(db_dir)
    return ChromaStore(db_dir)


class ChromaStore:
    def __init__(self, db_dir: Path):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(db_dir))
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Segment], embeddings: list[list[float]]) -> None:
        # ids must stay unique across separate ingest runs
        offset = self._collection.count()
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=[f"doc_{offset + start + i}" for i in range(end - start)],
                documents=[chunk.text for chunk in chunks[start:end]],
                metadatas=[chunk.metadata for chunk in chunks[start:end]],
                embeddings=embeddings[start:end],
            )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"document": document, "metadata": dict(metadata), "distance": distance})
        return hits

    def count(self) -> int:
        return self._collection.count()


class NumpyStore:
    def __init__(self, db_dir: Path):
        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_path = self._db_dir / "chunks.jsonl"
        self._vectors_path = self._db_dir / "vectors.npy"

    def add(self, chunks: list[Segment], embeddings: list[list[float]]) -> None:
        matrix = np.array(embeddings, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        with self._chunks_path.open("a", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps({"text": chunk.text, "metadata": chunk.metadata}, ensure_ascii=False) + "\n")
        if self._vectors_path.exists():
            old = np.load(self._vectors_path)
            matrix = np.vstack([old, matrix])
        np.save(self._vectors_path, matrix)

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        if not self._chunks_path.exists():
            return []
        chunks = [
            json.loads(line)
            for line in self._chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(self._vectors_path)
        query_vec = np.array(embedding, dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec) + 1e-12
        scores = vectors @ query_vec
        top_k = min(top_k, len(chunks))
        order = np.argpartition(-scores, top_k - 1)[:top_k]
        order = order[np.argsort(-scores[order])]
        return [
            {"document": chunks[i]["text"], "metadata": chunks[i]["metadata"], "distance": float(1 - scores[i])}
            for i in order
        ]

    def count(self) -> int:
        if not self._chunks_path.exists():
            return 0
        return sum(1 for line in self._chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
