import os
import sys
from pathlib import Path
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastai.learner import Learner
from fastai.metrics import rmse, mae
from fastai.callback.schedule import fit_one_cycle
from fastai.data.core import DataLoaders
from fastai.losses import MSELossFlat
from fastai.optimizer import Adam
from fastai.callback.core import Callback, CancelBatchException
from fastprogress.fastprogress import progress_bar
progress_bar.NO_BAR = True

from collections import defaultdict
from models.shared_utils import PAD_IDX
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SubjectDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'book_subjects': torch.tensor(row['book_subjects'], dtype=torch.long),
            'fav_subjects': torch.tensor(row['fav_subjects'], dtype=torch.long),
        }, torch.tensor(row['rating'], dtype=torch.float32)


class PerDimAttentionModel(nn.Module):
    def __init__(self, n_users, n_items, n_subjects, emb_dim=16, dropout=0.3):
        super().__init__()
        self.shared_subj_emb = nn.Embedding(n_subjects, emb_dim, padding_idx=PAD_IDX)
        self.attn_weight = nn.Parameter(torch.empty(emb_dim))
        self.attn_bias = nn.Parameter(torch.empty(emb_dim))
        nn.init.xavier_uniform_(self.attn_weight.unsqueeze(0))
        nn.init.zeros_(self.attn_bias)
        self.drop = nn.Dropout(dropout)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

    def attention_pool(self, indices):
        embs = self.shared_subj_emb(indices)  # [B, L, D]
        scores = (embs * self.attn_weight) + self.attn_bias  # [B, L, D]
        scores = scores.sum(dim=-1)  # [B, L]

        mask = (indices != PAD_IDX)
        has_real = mask.any(dim=1)
        for i in range(len(mask)):
            if not has_real[i]:
                mask[i, 0] = True

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B, L]
        pooled = (embs * attn.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return self.drop(pooled)

    def forward(self, batch):
        u = batch['user_idx']
        i = batch['item_idx']
        u_sub = batch['fav_subjects']
        i_sub = batch['book_subjects']

        u_emb = self.attention_pool(u_sub)
        i_emb = self.attention_pool(i_sub)
        dot = (u_emb * i_emb).sum(dim=1)

        return dot + self.user_bias(u).squeeze() + self.item_bias(i).squeeze() + self.global_bias

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "models" / "training" / "data"

def load_training_data_from_pickle(pad_to=5):
    print("📦 Loading .pkl data from:", DATA_DIR)

    interactions = pd.read_pickle(DATA_DIR / "interactions.pkl")
    user_fav_df = pd.read_pickle(DATA_DIR / "user_fav_subjects.pkl")
    book_subj_df = pd.read_pickle(DATA_DIR / "book_subjects.pkl")

    interactions = interactions[interactions["rating"].notnull()].copy()
    rating_counts = interactions["user_id"].value_counts()
    interactions["is_warm"] = interactions["user_id"].map(lambda uid: rating_counts.get(uid, 0) >= 10)

    user_fav = defaultdict(list)
    for row in user_fav_df.itertuples(index=False):
        user_fav[row.user_id].append(row.subject_idx)

    book_subj = defaultdict(list)
    for row in book_subj_df.itertuples(index=False):
        book_subj[row.item_idx].append(row.subject_idx)

    rows = []
    for row in interactions.itertuples(index=False):
        if not row.is_warm:
            continue

        book_subjs = book_subj.get(row.item_idx, [])
        if not book_subjs or all(s == PAD_IDX for s in book_subjs):
            continue

        fav_subjs = user_fav.get(row.user_id, [])
        if not fav_subjs:
            fav_subjs_padded = [PAD_IDX] * pad_to
        else:
            fav_subjs_padded = fav_subjs[:pad_to] + [PAD_IDX] * max(0, pad_to - len(fav_subjs))

        rows.append({
            "user_idx": row.user_id,
            "item_idx": row.item_idx,
            "rating": float(row.rating),
            "fav_subjects": fav_subjs_padded,
            "book_subjects": book_subjs
        })

    return rows


def main():
    rows = load_training_data_from_pickle()
    if not rows:
        print("❌ No valid training data found.")
        return

    print(f"✅ Loaded {len(rows)} training samples")
    max_len = max(len(r['book_subjects']) for r in rows)
    for r in rows:
        padded = r['book_subjects'] + [PAD_IDX] * (max_len - len(r['book_subjects']))
        r['book_subjects'] = padded

    ds = SubjectDataset(rows)
    cut = int(0.9 * len(ds))
    train_ds, valid_ds = torch.utils.data.random_split(ds, [cut, len(ds)-cut])
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=512, device=device)

    n_users = max(r['user_idx'] for r in rows) + 1
    n_items = max(r['item_idx'] for r in rows) + 1
    all_subjs = set(s for r in rows for s in r['book_subjects'] + r['fav_subjects'])
    n_subjects = max(all_subjs) + 1

    model = PerDimAttentionModel(n_users, n_items, n_subjects, emb_dim=16).to(device)

    learn = Learner(
        dls, model,
        loss_func=MSELossFlat(),
        metrics=[rmse, mae],
        wd=0.05,
        opt_func=Adam,
    )

    from fastai.callback.progress import ProgressCallback
    learn.remove_cbs(ProgressCallback)

    print("🚀 Starting training...")
    learn.fit_one_cycle(5, lr_max=3e-2)

    state = {
        'subject_embs': model.shared_subj_emb.state_dict(),
        'attn_weight': model.attn_weight.detach().cpu(),
        'attn_bias': model.attn_bias.detach().cpu(),
    }

    os.makedirs(REPO_ROOT / "models/data", exist_ok=True)
    torch.save(state, REPO_ROOT / "models/data/subject_attention_components_perdim.pth")
    print(f"✅ Saved to {REPO_ROOT}/models/data/subject_attention_components_perdim.pth")

if __name__ == "__main__":
    main()

