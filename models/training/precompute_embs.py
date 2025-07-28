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

from app.database import SessionLocal
from app.table_models import Book, BookSubject
from models.shared_utils import load_attention_components, attention_pool, batched_attention_pool, PAD_IDX

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load trained embedding + attention weights
print("📦 Loading subject embedding and attention components...")
subject_emb, attn_weight, attn_bias = load_attention_components("models/subject_attention_components.pth")
subject_emb = subject_emb.to(DEVICE)
attn_weight = attn_weight.to(DEVICE)
attn_bias = attn_bias.to(DEVICE)

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
pooled_embs = batched_attention_pool(subject_lists, subject_emb, attn_weight, attn_bias, batch_size=512)
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
