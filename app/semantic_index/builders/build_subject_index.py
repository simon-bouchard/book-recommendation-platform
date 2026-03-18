# app/semantic_index/builders/build_subject_index.py
"""
Build a FAISS flat-IP subject name index for semantic subject search.

Embeds all subject names from the llm_subjects table using the same
sentence-transformer model as the book index. Outputs:
  subjects.faiss        - IndexFlatIP (cosine via normalized vectors)
  subject_ids.npy       - int64 array mapping FAISS row -> llm_subject_idx
  subject_names.json    - {str(llm_subject_idx): {"name": str, "count": int}}

Usage:
  python -m app.semantic_index.builders.build_subject_index
  python -m app.semantic_index.builders.build_subject_index --output models/artifacts/semantic_indexes/subjects_v1
  python -m app.semantic_index.builders.build_subject_index --model sentence-transformers/all-MiniLM-L6-v2
"""

from pathlib import Path
import sys
import json
import argparse

import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sqlalchemy import func

from app.database import SessionLocal
from app.table_models import Subject, BookSubject

_DEFAULT_OUTPUT = "models/artifacts/semantic_indexes/subjects_v1"
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def fetch_subjects() -> list[tuple[int, str, int]]:
    """Return [(subject_idx, subject_name, book_count), ...] for all non-null subjects."""
    with SessionLocal() as db:
        rows = (
            db.query(
                Subject.subject_idx,
                Subject.subject,
                func.count(BookSubject.item_idx).label("count"),
            )
            .outerjoin(BookSubject, BookSubject.subject_idx == Subject.subject_idx)
            .group_by(Subject.subject_idx, Subject.subject)
            .all()
        )
    return [(int(idx), name, int(count)) for idx, name, count in rows if name]


def build_subject_index(
    subjects: list[tuple[int, str, int]],
    model_name: str,
    output_dir: str,
) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    ids, names, counts = zip(*subjects)
    ids = list(ids)
    names = list(names)
    counts = list(counts)

    print(f"Embedding {len(names)} subjects...")
    embeddings = model.encode(
        names,
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    d = embeddings.shape[1]
    print(f"Building IndexFlatIP (dim={d})...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "subjects.faiss"))
    np.save(out / "subject_ids.npy", np.array(ids, dtype=np.int64))
    with open(out / "subject_names.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(sid): {"name": name, "count": count} for sid, name, count in zip(ids, names, counts)},
            f,
            ensure_ascii=False,
        )

    print(f"\nSaved to {output_dir}/")
    print(f"  subjects.faiss     {index.ntotal} vectors, dim={d}")
    print(f"  subject_ids.npy    {len(ids)} entries")
    print(f"  subject_names.json {len(ids)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build subject semantic search index")
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output directory (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Sentence-transformer model name (default: {_DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    subjects = fetch_subjects()
    print(f"Fetched {len(subjects)} subjects from database")
    build_subject_index(subjects, args.model, args.output)
