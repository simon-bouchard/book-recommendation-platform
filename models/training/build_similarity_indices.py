import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(os.path.abspath(str(REPO_ROOT)))

from models.core.paths import PATHS
from models.data.loaders import normalize_embeddings
from models.infrastructure.similarity_index import SimilarityIndex


def _build_subject_index(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(PATHS.staging_dir / "embeddings" / "book_subject_embeddings.npy")
    with open(PATHS.staging_dir / "embeddings" / "book_subject_ids.json") as f:
        ids = json.load(f)

    embeddings = normalize_embeddings(embeddings)

    index = SimilarityIndex(
        embeddings=embeddings,
        ids=ids,
        normalize=False,
        use_hnsw=True,
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=200,
    )
    index.save(output_dir)
    print(f"Subject index saved: {index.num_total} items, {index.num_candidates} candidates")


def _build_als_index(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    book_factors = np.load(PATHS.staging_dir / "embeddings" / "book_als_factors.npy")
    with open(PATHS.staging_dir / "embeddings" / "book_als_ids.json") as f:
        book_ids = json.load(f)

    books = pd.read_pickle(PATHS.staging_data_dir / "books.pkl").set_index("item_idx")

    index = SimilarityIndex.create_filtered_index(
        embeddings=book_factors,
        ids=book_ids,
        metadata=books,
        min_rating_count=10,
        normalize=True,
    )
    index.save(output_dir)
    print(f"ALS index saved: {index.num_total} items, {index.num_candidates} candidates")


def main():
    PATHS.ensure_staging_dirs()
    staging_sim = PATHS.staging_dir / "similarity"

    print("Building subject similarity index...")
    _build_subject_index(staging_sim / "subject")

    print("Building ALS similarity index...")
    _build_als_index(staging_sim / "als")


if __name__ == "__main__":
    main()
