import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastai.learner import Learner
from fastai.metrics import rmse
from fastai.callback.schedule import fit_one_cycle
from fastai.data.core import DataLoaders
from fastai.losses import MSELossFlat
from fastai.optimizer import Adam
from fastai.callback.core import Callback, CancelBatchException

from collections import defaultdict
from sqlalchemy.orm import joinedload
from app.database import SessionLocal
from app.table_models import Interaction, BookSubject, UserFavSubject
from models.shared_utils import PAD_IDX
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------
# Dataset
# ---------------------
class SubjectDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'book_subjects': torch.tensor(row['book_subjects'], dtype=torch.long),
            'fav_subjects': torch.tensor(row['fav_subjects'], dtype=torch.long),
        }, torch.tensor(row['rating'], dtype=torch.float32)


# ---------------------
# Model
# ---------------------
class SubjectDotModel(nn.Module):
    def __init__(self, n_users, n_items, n_subjects, emb_dim=16, dropout=0.3):
        super().__init__()
        self.shared_subj_emb = nn.Embedding(n_subjects, emb_dim, padding_idx=PAD_IDX)
        self.subject_attn = nn.Linear(emb_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

    def attention_pool(self, indices):
        embs = self.shared_subj_emb(indices)
        scores = self.subject_attn(embs).squeeze(-1)

        mask = (indices != PAD_IDX) 
        has_real_subjects = mask.any(dim=1)  

        # Create safe_mask that un-masks one PAD if all are PADs
        safe_mask = mask.clone()
        for i in range(len(safe_mask)):
            if not has_real_subjects[i]:
                safe_mask[i, 0] = True 

        scores = scores.masked_fill(~safe_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)

        pooled = (embs * weights.unsqueeze(-1)).sum(dim=1)
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


# ---------------------
# Data loading from SQL
# ---------------------
def fetch_training_data(pad_to=10):
    db = SessionLocal()

    # Get user → subjects
    user_to_subj = defaultdict(list)
    for row in db.query(UserFavSubject.user_id, UserFavSubject.subject_idx):
        user_to_subj[row.user_id].append(row.subject_idx)

    # Get book → subjects
    book_to_subj = defaultdict(list)
    for row in db.query(BookSubject.item_idx, BookSubject.subject_idx):
        book_to_subj[row.item_idx].append(row.subject_idx)

    # Count ratings per user
    user_rating_counts = defaultdict(int)
    for row in db.query(Interaction.user_id).filter(Interaction.rating != None):
        user_rating_counts[row.user_id] += 1

    # Get interactions
    rows = []
    for row in db.query(Interaction.user_id, Interaction.item_idx, Interaction.rating).filter(Interaction.rating != None):
        # Filter: Only keep warm users
        if user_rating_counts[row.user_id] < 10:
            continue

        book_subjs = book_to_subj.get(row.item_idx, [])
        user_subjs = user_to_subj.get(row.user_id, [])

        if not book_subjs or not user_subjs:
            continue

        rows.append({
            'user_idx': row.user_id,
            'item_idx': row.item_idx,
            'rating': float(row.rating),
            'book_subjects': book_subjs,
            'fav_subjects': user_subjs[:5] + [PAD_IDX] * max(0, 5 - len(user_subjs))  # still pad favs
        })

    db.close()
    return rows

class GradientAccumulation(Callback):
    def __init__(self, n_acc=4): self.n_acc = n_acc
    def before_fit(self): self.acc_count = 0
    def before_backward(self): self.learn.loss_grad /= self.n_acc
    def after_backward(self):
        self.acc_count += 1
        if self.acc_count % self.n_acc == 0:
            self.learn.opt.step()
            self.learn.opt.zero_grad()
        else:
            raise CancelBatchException()
    def after_fit(self): self.learn.opt.zero_grad()

# ---------------------
# Train and save
# ---------------------
def main():
    rows = fetch_training_data()
    if not rows:
        print("❌ No valid training data found.")
        return

    print(f"✅ Loaded {len(rows)} training samples")
    
    max_len = max(len(r['book_subjects']) for r in rows)

    for r in rows:
        padded = r['book_subjects'] + [PAD_IDX] * (max_len - len(r['book_subjects']))
        r['book_subjects'] = padded

    # Create dataset and dataloaders
    ds = SubjectDataset(rows)
    cut = int(0.9 * len(ds))
    train_ds, valid_ds = torch.utils.data.random_split(ds, [cut, len(ds)-cut])
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=128, device=device)

    # Count ranges
    n_users = max(r['user_idx'] for r in rows) + 1
    n_items = max(r['item_idx'] for r in rows) + 1
    all_subjs = set(s for r in rows for s in r['book_subjects'] + r['fav_subjects'])
    n_subjects = max(all_subjs) + 1

    model = SubjectDotModel(n_users=n_users, n_items=n_items, n_subjects=n_subjects, emb_dim=16).to(device)

    learn = Learner(
        dls, model,
        loss_func=MSELossFlat(),
        metrics=[rmse],
        wd=0.05,
        opt_func=Adam,
        #cbs=[GradientAccumulation(n_acc=4)]
    )

    print("🚀 Starting training...")
    learn.fit_one_cycle(5, lr_max=3e-2)

    # Save components
    state = {
        'subject_embs': model.shared_subj_emb.state_dict(),
        'attn_weight': model.subject_attn.weight.detach().cpu(),
        'attn_bias': model.subject_attn.bias.detach().cpu(),
    }

    os.makedirs("models", exist_ok=True)
    torch.save(state, "models/subject_attention_components.pth")
    print("✅ Saved to models/subject_attention_components.pth")


if __name__ == "__main__":
    main()
