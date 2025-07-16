# models/precompute_item_embs.py

import os
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

PAD_IDX = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# Load trained embedding + attention weights
# ----------------------------
state_path = "models/subject_attention_components.pth"
print(f"📦 Loading: {state_path}")
state = torch.load(state_path, map_location=DEVICE)

# Subject embedding layer
emb_weight = state["subject_embs"]["weight"]
num_embeddings, emb_dim = emb_weight.shape
subject_emb = nn.Embedding(num_embeddings, emb_dim, padding_idx=PAD_IDX)
subject_emb.load_state_dict({"weight": emb_weight})
subject_emb.to(DEVICE)
subject_emb.weight.requires_grad_(False)

# Attention components
attn_weight = state["attn_weight"].to(DEVICE)
attn_bias = state["attn_bias"].to(DEVICE)

# ----------------------------
# Attention pooling
# ----------------------------
def attention_pool(indices_list):
    max_len = max((len(lst) for lst in indices_list), default=1)
    padded = [lst + [PAD_IDX] * (max_len - len(lst)) for lst in indices_list]
    idx_tensor = torch.tensor(padded, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        emb = subject_emb(idx_tensor)
        mask = (idx_tensor != PAD_IDX)

        logits = (emb @ attn_weight.T).squeeze(-1) + attn_bias
        logits[~mask] = float('-inf')

        weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * emb).sum(dim=1)

    return pooled.cpu().numpy()

def batched_attention_pool(indices_list, batch_size=1024):
    all_outputs = []
    for i in range(0, len(indices_list), batch_size):
        batch = indices_list[i:i+batch_size]
        pooled = attention_pool(batch)
        all_outputs.append(pooled)
    return np.concatenate(all_outputs, axis=0)

# ----------------------------
# Load books + subjects from SQL
# ----------------------------
print("🔄 Loading books and subjects from SQL...")
db = SessionLocal()

book_to_subj = defaultdict(list)
for row in db.query(BookSubject.item_idx, BookSubject.subject_idx):
    book_to_subj[row.item_idx].append(row.subject_idx)

books = db.query(Book.item_idx).all()
db.close()

print(f"📘 Found {len(books)} books")

book_ids = []
subject_lists = []

for row in books:
    item_idx = row.item_idx
    subjects = book_to_subj.get(item_idx, [])
    if not subjects or all(s == PAD_IDX for s in subjects):
        continue
    book_ids.append(item_idx)
    subject_lists.append(subjects)

print(f"✅ Books with valid subjects: {len(book_ids)}")

# ----------------------------
# Compute pooled embeddings
# ----------------------------
print("🧠 Computing pooled subject embeddings...")
pooled_embs = batched_attention_pool(subject_lists, batch_size=512)
print(f"📐 Shape: {pooled_embs.shape}")

# ----------------------------
# Save outputs
# ----------------------------
os.makedirs("models", exist_ok=True)
np.save("models/book_embs.npy", pooled_embs)

with open("models/book_ids.json", "w") as f:
    json.dump(book_ids, f)

print("✅ Saved:")
print("   - models/book_embs.npy")
print("   - models/book_ids.json")