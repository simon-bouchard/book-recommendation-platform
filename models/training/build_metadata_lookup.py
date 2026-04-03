import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(os.path.abspath(os.path.join(REPO_ROOT)))

from models.core.paths import PATHS
from models.data.loaders import get_item_idx_to_row
from models.infrastructure.metadata_enrichment import build_lookup


def main():
    print("Loading files...")
    books = pd.read_pickle(PATHS.staging_data_dir / "books.pkl").set_index("item_idx")

    bayesian_scores = np.load(PATHS.staging_dir / "scoring" / "bayesian_scores.npy")
    bayesian_scores = np.nan_to_num(bayesian_scores, nan=0.0, posinf=0.0, neginf=0.0)

    with open(PATHS.staging_dir / "embeddings" / "book_subject_ids.json") as f:
        book_ids = json.load(f)

    idx_to_row = get_item_idx_to_row(book_ids)
    books["bayes"] = books.index.to_series().map(
        lambda idx: float(bayesian_scores[idx_to_row.get(int(idx), -1)])
        if idx_to_row.get(int(idx)) is not None
        else float("-inf")
    )
    books = books.sort_values(["bayes", "title"], ascending=[False, True], kind="mergesort")

    print(f"Building lookup dict for {len(books)} books...")
    lookup = build_lookup(books)

    output_path = PATHS.staging_data_dir / "book_lookup.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(lookup, f)

    print(f"Saved: {output_path} ({len(lookup)} entries)")


if __name__ == "__main__":
    main()
