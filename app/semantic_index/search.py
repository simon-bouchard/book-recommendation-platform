from pathlib import Path
import numpy as np
import faiss
import json

class SemanticSearcher:
    def __init__(self, dir_path: str, embedder):
        self.dir = Path(dir_path)
        self.index = faiss.read_index(str(self.dir / "semantic.faiss"))
        self.ids = np.load(self.dir / "semantic_ids.npy")
        with open(self.dir / "semantic_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embedder = embedder

    def search(self, query: str, top_k: int = 10):
        qv = self.embedder([query]).astype("float32")
        D, I = self.index.search(qv, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: 
                continue
            bid = int(self.ids[idx])
            results.append({"book_id": bid, "score": float(-dist), "meta": self.meta[idx]})
        return results
