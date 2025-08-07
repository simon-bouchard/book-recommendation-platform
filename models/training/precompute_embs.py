import os
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.shared_utils import PAD_IDX, ModelStore

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar")

# ----------------------------
# Load .pkl files
# ----------------------------
print("📄 Loading book and subject mappings...")
books = pd.read_pickle("models/training/data/books.pkl")
book_subjects = pd.read_pickle("models/training/data/book_subjects.pkl")

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
pooler = ModelStore().get_attention_strategy(ATTN_STRATEGY)
pooled_embs = pooler(subject_lists).cpu().numpy()
print(f"📐 Shape: {pooled_embs.shape}")

# ----------------------------
# Save outputs
# ----------------------------
os.makedirs("models", exist_ok=True)
np.save("models/data/book_embs.npy", pooled_embs)

with open("models/data/book_ids.json", "w") as f:
    json.dump(book_ids, f)

print("✅ Saved:")
print("   - models/data/book_embs.npy")
print("   - models/data/book_ids.json")
