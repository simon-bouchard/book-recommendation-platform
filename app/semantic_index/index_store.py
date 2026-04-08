import json
from pathlib import Path

import numpy as np


class IndexStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, faiss_path: Path, ids_path: Path, meta_path: Path, faiss_index, ids, meta):
        import faiss

        faiss.write_index(faiss_index, str(faiss_path))
        np.save(ids_path, np.asarray(ids, dtype=np.int64))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, faiss_path: Path, ids_path: Path, meta_path: Path):
        import faiss

        index = faiss.read_index(str(faiss_path))
        ids = np.load(ids_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return index, ids, meta
