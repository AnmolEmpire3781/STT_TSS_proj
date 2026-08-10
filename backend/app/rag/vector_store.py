import hashlib
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from .chunking import Chunk

class VectorStore:
    def __init__(self):
        s = get_settings()
        self.settings = s
        self.model = SentenceTransformer(s.embedding_model)
        self.client = chromadb.PersistentClient(path=str(s.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=s.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        arr = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return arr.tolist()

    def reset(self):
        try:
            self.client.delete_collection(self.settings.chroma_collection)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        ids, metas = [], []
        for c in chunks:
            raw_id = f"{c.source}:{c.page}:{c.index}:{c.text}".encode("utf-8")
            ids.append(hashlib.sha1(raw_id).hexdigest())
            metas.append({"source": c.source, "page": c.page or -1, "chunk_index": c.index})
        self.collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        return len(chunks)

    def query(self, text: str, top_k: int, min_score: float) -> list[dict]:
        if self.collection.count() == 0:
            return []
        q = self._embed([text])[0]
        res = self.collection.query(query_embeddings=[q], n_results=top_k)
        out = []
        for i, doc in enumerate(res.get("documents", [[]])[0]):
            distance = float(res.get("distances", [[]])[0][i])
            score = 1.0 - distance
            meta = res.get("metadatas", [[]])[0][i] or {}
            if score < min_score:
                continue
            out.append({
                "id": res.get("ids", [[]])[0][i],
                "source": meta.get("source", "unknown"),
                "page": None if meta.get("page", -1) == -1 else int(meta.get("page")),
                "score": score,
                "text": doc,
            })
        return out

@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
