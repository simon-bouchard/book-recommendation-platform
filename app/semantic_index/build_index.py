from pathlib import Path
import json
import numpy as np
import faiss

# Expect enrichment_v1.jsonl created by runner.py
def yield_texts(enrichment_jsonl: Path, books_csv: Path):
    # Minimal: we only need title/author; adjust if you want more fields
    import csv
    id2meta = {}
    with open(books_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bid = r.get("work_id") or r.get("id")
            id2meta[str(bid)] = {"title": r.get("title",""), "author": r.get("author","")}
    with open(enrichment_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec: 
                continue
            bid = str(rec["book_id"])
            meta = id2meta.get(bid, {"title":"", "author":""})
            subjects = ", ".join(rec.get("subjects", []))
            # map tone_ids to slugs if you want readable text; optional for embeddings
            tone_ids = rec.get("tone_ids", [])
            vibe = rec.get("vibe","")
            text = f'{meta["title"]} — {meta["author"]} | subjects: {subjects} | tones: {",".join(map(str,tone_ids))} | vibe: {vibe}'
            yield int(rec["book_id"]), text, {"title": meta["title"], "author": meta["author"], "tone_ids": tone_ids, "subjects": rec.get("subjects", []), "vibe": vibe}

def embed_texts(texts, embedder):
    # embedder: callable list[str] -> np.ndarray[float32] (n, d)
    batch, ids, metas = [], [], []
    vecs = []
    for bid, txt, meta in texts:
        ids.append(bid); metas.append(meta); batch.append(txt)
        if len(batch) >= 256:
            v = embedder(batch)
            vecs.append(v)
            batch.clear()
    if batch:
        vecs.append(embedder(batch))
    return np.concatenate(vecs, axis=0), np.array(ids, dtype=np.int64), metas

def build_faiss(embeds: np.ndarray):
    d = embeds.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 80
    index.add(embeds.astype("float32"))
    return index

def main(enrichment_path: str, books_csv: str, out_dir: str, embedder):
    texts = list(yield_texts(Path(enrichment_path), Path(books_csv)))
    embeds, ids, metas = embed_texts(texts, embedder)
    index = build_faiss(embeds)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    faiss_path = out / "semantic.faiss"
    ids_path = out / "semantic_ids.npy"
    meta_path = out / "semantic_meta.json"
    from .index_store import IndexStore
    IndexStore(out).save(faiss_path, ids_path, meta_path, index, ids, metas)
