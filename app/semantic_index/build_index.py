# app/semantic_index/build_index.py
import json
from pathlib import Path

import faiss
import numpy as np

from app.database import SessionLocal
from app.table_models import Author, Book

from .index_store import IndexStore


def yield_texts(enrichment_jsonl: Path):
    """
    Generator that yields (item_idx, text_for_embedding, metadata_dict) tuples.

    - Reads enrichment JSONL created by the enrichment runner/backfill.
    - Expects `book_id` values to be integers (Book.item_idx).
    - Fetches title and author from the database for each item_idx.
    - Constructs a concatenated text string (title, author, subjects, tones, vibe)
      that will be embedded and indexed.
    """
    ids = set()
    recs = []
    # Collect enrichment records and unique item_idx values
    with open(enrichment_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "error" in obj:
                continue  # skip error rows
            bid = obj.get("book_id")
            if not isinstance(bid, int):
                continue  # enforce integer IDs only
            ids.add(bid)
            recs.append(obj)

    # Bulk-fetch book metadata for all required item_idx values
    by_item = {}
    with SessionLocal() as db:
        q = (
            db.query(Book, Author)
            .outerjoin(Author, Book.author_idx == Author.author_idx)
            .filter(Book.item_idx.in_(ids))
        )
        for b, a in q:
            by_item[int(b.item_idx)] = (
                b.title or "",
                (a.name if a else "") or "",
            )

    # Yield tuples for embedding
    for rec in recs:
        bid = int(rec["book_id"])
        title, author = by_item.get(bid, ("", ""))
        subjects = ", ".join(rec.get("subjects", []))
        tone_ids = rec.get("tone_ids", [])
        vibe = rec.get("vibe", "")
        # Text string used for embeddings; compact but information-rich
        text = (
            f"{title} — {author} | "
            f"subjects: {subjects} | "
            f"tones: {','.join(map(str, tone_ids))} | "
            f"vibe: {vibe}"
        )
        meta = {
            "title": title,
            "author": author,
            "tone_ids": tone_ids,
            "subjects": rec.get("subjects", []),
            "vibe": vibe,
        }
        yield bid, text, meta


def embed_texts(texts, embedder):
    """
    Embed all book texts into dense vectors.

    - Uses batching (size 256) for efficiency.
    - Returns (embeddings array, ids array[int64], metadata list).
    """
    batch, ids, metas, vecs = [], [], [], []
    for bid, txt, meta in texts:
        ids.append(bid)
        metas.append(meta)
        batch.append(txt)
        if len(batch) >= 256:
            vecs.append(embedder(batch))
            batch.clear()
    if batch:
        vecs.append(embedder(batch))
    return np.concatenate(vecs, axis=0), np.array(ids, dtype=np.int64), metas


def build_faiss(embeds: np.ndarray):
    """
    Build a FAISS HNSW index over the embedding vectors.

    - HNSW provides high-recall approximate nearest neighbor search.
    - efConstruction tuned to 80 for a balance of recall vs. memory.
    """
    d = embeds.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 80
    index.add(embeds.astype("float32"))
    return index


def main(enrichment_path: str, out_dir: str, embedder):
    """
    Build the semantic search index from enrichment outputs.

    Steps:
      1. Read enrichment JSONL (expects item_idx as book_id).
      2. Fetch metadata from DB (title, author).
      3. Construct text inputs and embed them.
      4. Build FAISS index and save with ids + metadata.
    """
    texts = list(yield_texts(Path(enrichment_path)))
    if not texts:
        raise RuntimeError("No valid enrichment records found.")

    embeds, ids, metas = embed_texts(texts, embedder)
    index = build_faiss(embeds)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    IndexStore(out).save(
        out / "semantic.faiss",
        out / "semantic_ids.npy",
        out / "semantic_meta.json",
        index,
        ids,
        metas,
    )
