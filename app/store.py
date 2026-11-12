"""Vector store wrapper around ChromaDB.

Every chunk is tagged with a customer_id, and every search is filtered by it,
so one customer can never retrieve another customer's documents.
"""
import hashlib
from pathlib import Path

import chromadb


class Store:
    def __init__(self, path: Path, embedding_function=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        kwargs = {"name": "documents", "metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.col = self.client.get_or_create_collection(**kwargs)

    @staticmethod
    def _chunk_id(customer_id: str, source: str, index: int) -> str:
        raw = f"{customer_id}|{source}|{index}".encode()
        return hashlib.sha1(raw).hexdigest()

    def replace_document(self, customer_id: str, source: str, chunks: list[dict]) -> int:
        """Delete any old chunks of this document, then add the new ones.

        Each chunk is {"text": str, "page": int}. Running ingestion twice on the
        same file therefore re-indexes it instead of duplicating it.
        """
        self.col.delete(where={"$and": [{"customer_id": customer_id}, {"source": source}]})
        if not chunks:
            return 0
        self.col.add(
            ids=[self._chunk_id(customer_id, source, i) for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {"customer_id": customer_id, "source": source, "page": int(c.get("page", 0))}
                for c in chunks
            ],
        )
        return len(chunks)

    def delete_customer(self, customer_id: str) -> None:
        self.col.delete(where={"customer_id": customer_id})

    def search(self, customer_id: str, query: str, k: int) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_texts=[query],
            n_results=min(k, self.col.count()),
            where={"customer_id": customer_id},
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                {
                    "text": text,
                    "source": meta["source"],
                    "page": meta.get("page", 0),
                    "distance": float(dist),
                }
            )
        return hits
