# models/infrastructure/subject_searcher.py
"""
Subject name semantic searcher.

Loads a pre-built FAISS flat-IP index of subject name embeddings and searches
for the nearest neighbours of a free-text phrase. Vectors are L2-normalized
at index build time; query vectors are normalized before search so that
inner product equals cosine similarity.
"""

from pathlib import Path
import json

import numpy as np
import faiss


class SubjectSearcher:
    def __init__(self, dir_path: str, embedder):
        self.dir = Path(dir_path)
        self.index = faiss.read_index(str(self.dir / "subjects.faiss"))
        self.ids = np.load(self.dir / "subject_ids.npy")
        with open(self.dir / "subject_names.json", encoding="utf-8") as f:
            self._names: dict[str, str] = json.load(f)
        self.embedder = embedder

    def search(self, phrase: str, top_k: int = 5) -> list[dict]:
        qv = self.embedder([phrase]).astype("float32")
        norm = np.linalg.norm(qv, axis=1, keepdims=True)
        qv = qv / np.where(norm == 0, 1.0, norm)
        D, I = self.index.search(qv, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            subject_idx = int(self.ids[idx])
            entry = self._names.get(str(subject_idx), {})
            results.append({
                "subject_idx": subject_idx,
                "subject_name": entry.get("name", ""),
                "count": entry.get("count", 0),
                "score": float(score),
            })
        return results
