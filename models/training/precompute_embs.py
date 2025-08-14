import os
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.shared_utils import PAD_IDX, ModelStore

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "models" / "training" / "data"
MODELS_DIR = ROOT / "models" / "data"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar")

# ----------------------------
# Load .pkl files
# ----------------------------
print("📄 Loading book and subject mappings...")
books = pd.read_pickle(DATA_DIR / "books.pkl")
book_subjects = pd.read_pickle(DATA_DIR / "book_subjects.pkl")

book_to_subjects = defaultdict(list)
for row in book_subjects.itertuples():
    book_to_subjects[row.item_idx].append(row.subject_idx)

book_ids = []
subject_lists = []

for row in books.itertuples():
    subjects = book_to_subjects.get(row.item_idx, [])
    if not subjects or all(s == PAD_IDX for s in subjects):
        continue
    book_ids.append(row.item_idx)
    subject_lists.append(subjects)

print(f"✅ Books with valid subjects: {len(book_ids)}")

# ----------------------------
# Compute pooled embeddings
# ----------------------------
print("🧠 Computing pooled subject embeddings...")
BATCH_SIZE = 1024
all_embs = []
pooler = ModelStore().get_attention_strategy(ATTN_STRATEGY)

with torch.no_grad():
    for start in range(0, len(subject_lists), BATCH_SIZE):
        batch = subject_lists[start:start+BATCH_SIZE]
        embs = pooler(batch).cpu().numpy()
        all_embs.append(embs)

pooled_embs = np.vstack(all_embs)

print(f"📐 Shape: {pooled_embs.shape}")

# ----------------------------
# Save outputs
# ----------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
np.save(MODELS_DIR / "book_embs.npy", pooled_embs)

with open(f"{MODELS_DIR}/book_ids.json", "w") as f:
    json.dump(book_ids, f)

print("✅ Saved:")
print(f"   - {MODELS_DIR}/book_embs.npy")
print(f"   - {MODELS_DIR}/book_ids.json")
