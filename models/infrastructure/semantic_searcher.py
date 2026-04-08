from pathlib import Path

import faiss
import numpy as np


class SemanticSearcher:
    def __init__(self, dir_path: str, embedder):
        self.dir = Path(dir_path)
        self.index = faiss.read_index(str(self.dir / "semantic.faiss"))
        self.ids = np.load(self.dir / "semantic_ids.npy")
        self.embedder = embedder

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        qv = self.embedder([query]).astype("float32")
        D, indices = self.index.search(qv, top_k)
        results = []
        for dist, idx in zip(D[0], indices[0]):
            if idx == -1:
                continue
            results.append({"item_idx": int(self.ids[idx]), "score": float(-dist)})
        return results
