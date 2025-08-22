import os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler
from collections import defaultdict

from models.shared_utils import PAD_IDX

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "models" / "training" / "data"
OUT_PATH = REPO_ROOT / "models" / "data" / "subject_attention_components_perdim.pth"

# ---------------------------
# Dataset
# ---------------------------
class SubjectDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "user_idx": torch.tensor(r["user_idx"], dtype=torch.long),
            "item_idx": torch.tensor(r["item_idx"], dtype=torch.long),
            "book_subjects": torch.tensor(r["book_subjects"], dtype=torch.long),
            "fav_subjects": torch.tensor(r["fav_subjects"], dtype=torch.long),
            "rating": torch.tensor(r["rating"], dtype=torch.float32),
        }

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
        if not row.is_warm:  # we keep your warm-user filter from the per-dim trainer
            continue

        book_subjs = book_subj.get(row.item_idx, [])
        if not book_subjs or all(s == PAD_IDX for s in book_subjs):
            continue

        fav_subjs = user_fav.get(row.user_id, [])
        fav_subjs_padded = (fav_subjs[:pad_to] + [PAD_IDX] * max(0, pad_to - len(fav_subjs))) if fav_subjs else [PAD_IDX] * pad_to

        rows.append({
            "user_idx": row.user_id,
            "item_idx": row.item_idx,
            "rating": float(row.rating),
            "fav_subjects": fav_subjs_padded,
            "book_subjects": book_subjs
        })

    # pad book_subjects to a common length (same as your current script)
    max_len = max(len(r["book_subjects"]) for r in rows)
    for r in rows:
        r["book_subjects"] = r["book_subjects"] + [PAD_IDX] * (max_len - len(r["book_subjects"]))
    return rows

# ---------------------------
# Model (per-dim attention)
# ---------------------------
class PerDimAttentionModel(nn.Module):
    def __init__(self, n_users, n_items, n_subjects, emb_dim=64, dropout=0.3, attn_tau=1.0):
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
        self.attn_tau = float(attn_tau)

    def attention_pool(self, indices):
        # indices: [B, L]
        embs = self.shared_subj_emb(indices)                     # [B, L, D]
        scores = (embs * self.attn_weight) + self.attn_bias      # [B, L, D]
        scores = scores.sum(dim=-1)                               # [B, L]
        mask = (indices != PAD_IDX)
        # ensure at least one token
        has_real = mask.any(dim=1)
        mask = mask.clone()
        mask[~has_real, 0] = True
        scores = scores.masked_fill(~mask, float("-inf"))         # [B, L]
        attn = torch.softmax(scores / self.attn_tau, dim=1)       # [B, L]
        pooled = (embs * attn.unsqueeze(-1)).sum(dim=1)           # [B, D]
        return self.drop(pooled)

    def forward(self, batch):
        u = batch["user_idx"]
        i = batch["item_idx"]
        u_emb = self.attention_pool(batch["fav_subjects"])
        i_emb = self.attention_pool(batch["book_subjects"])
        dot = (u_emb * i_emb).sum(dim=1)
        pred = dot + self.user_bias(u).squeeze() + self.item_bias(i).squeeze() + self.global_bias
        return pred, i_emb, u

# ---------------------------
# Contrastive utils
# ---------------------------
@torch.no_grad()
def _jaccard_pos_mask_from_indices(book_subjects, thresh_overlap=2):
    """
    book_subjects: LongTensor [B, L] (padded with PAD_IDX)
    Returns: BoolTensor [B, B] where mask[i,j] True if overlap>=thresh and i!=j
    """
    B, L = book_subjects.shape
    # turn into sets per row (loop is fine; done only per batch)
    sets = []
    for b in range(B):
        row = book_subjects[b].tolist()
        s = set(x for x in row if x != PAD_IDX)
        sets.append(s)
    M = torch.zeros(B, B, dtype=torch.bool, device=book_subjects.device)
    for i in range(B):
        si = sets[i]
        for j in range(i+1, B):
            if len(si & sets[j]) >= thresh_overlap:
                M[i, j] = True
                M[j, i] = True
    return M

def multi_positive_infonce(item_emb, user_idx, book_subjects,
                           t=0.07, use_jaccard=True, overlap_thresh=2):
    """
    item_emb: [B, D] (not normalized)
    user_idx: [B]
    book_subjects: [B, L]
    Returns: (loss, stats_dict)
    """
    B = item_emb.size(0)
    Z = F.normalize(item_emb, dim=1)         # cosine
    sim = (Z @ Z.t()) / t                    # [B,B]
    sim = sim - torch.eye(B, device=sim.device) * 1e9  # mask self

    same_user = user_idx.unsqueeze(1).eq(user_idx.unsqueeze(0))  # [B,B]
    same_user.fill_diagonal_(False)

    if use_jaccard:
        with torch.no_grad():
            subj_mask = _jaccard_pos_mask_from_indices(book_subjects, thresh_overlap=overlap_thresh)
    else:
        subj_mask = torch.zeros_like(same_user)

    pos_mask = (same_user | subj_mask)      # multi-positive
    valid = pos_mask.any(dim=1)             # anchors that have >=1 positive

    if not valid.any():
        # No positives in this batch: return 0 loss and empty stats (caller will skip)
        return torch.zeros([], device=item_emb.device), {
            "pct_with_positive": torch.tensor(0.0, device=item_emb.device),
            "pos_mean": torch.tensor(float("nan"), device=item_emb.device),
            "neg_mean": torch.tensor(float("nan"), device=item_emb.device),
        }

    num = torch.logsumexp(sim.masked_fill(~pos_mask, float("-inf")), dim=1)
    den = torch.logsumexp(sim, dim=1)
    loss_vec = -(num - den)[valid]
    loss = loss_vec.mean()

    # Diagnostics (don’t backprop through stats)
    with torch.no_grad():
        pos_vals = sim[pos_mask]
        neg_vals = sim[(~pos_mask) & (~torch.eye(B, device=sim.device, dtype=torch.bool))]
        stats = {
            "pct_with_positive": valid.float().mean() * 100.0,
            "pos_mean": pos_vals.mean() if pos_vals.numel() else torch.tensor(float("nan"), device=sim.device),
            "neg_mean": neg_vals.mean() if neg_vals.numel() else torch.tensor(float("nan"), device=sim.device),
        }

    return loss, stats

# ---------------------------
# Train
# ---------------------------
def main():
    rows = load_training_data_from_pickle()
    if not rows:
        print("❌ No valid training data found.")
        return

    ds = SubjectDataset(rows)
    # RandomSampler with replacement improves chance of repeated users per batch (for positives)
    sampler = RandomSampler(ds, replacement=True, num_samples=len(ds))
    dl = DataLoader(ds, batch_size=512, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    n_users = max(r["user_idx"] for r in rows) + 1
    n_items = max(r["item_idx"] for r in rows) + 1
    all_subjs = set(s for r in rows for s in r["book_subjects"] + r["fav_subjects"])
    n_subjects = max(all_subjs) + 1

    model = PerDimAttentionModel(n_users, n_items, n_subjects, emb_dim=64, dropout=0.1, attn_tau=1.0).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-2)
    epochs = 6
    lambda_contrast = 0.8
    lambda_mse = 0.2
    t_contrast = 0.07
    use_jaccard = True          # also treat high subject-overlap pairs as positives
    overlap_thresh = 2          # >=2 shared subjects

    mse_loss = nn.MSELoss()

	print("🚀 Starting contrastive+RMSE training...")
	for epoch in range(epochs):
		model.train()

		# running sums for epoch-level metrics
		sum_total, sum_contrast, sum_mse = 0.0, 0.0, 0.0
		sse, n_examples = 0.0, 0
		pos_count_pct, pos_mean_accum, neg_mean_accum, stat_steps = 0.0, 0.0, 0.0, 0

		for step, batch in enumerate(dl, 1):
			for k in batch:
				batch[k] = batch[k].to(device, non_blocking=True)

			opt.zero_grad(set_to_none=True)
			pred, i_emb, u_idx = model(batch)

			# --- supervised head ---
			mse = mse_loss(pred, batch["rating"])
			# also accumulate RMSE components over the whole epoch
			with torch.no_grad():
				se = (pred.detach() - batch["rating"])**2
				sse += se.sum().item()
				n_examples += se.numel()

			# --- contrastive head ---
			contrast, stats = multi_positive_infonce(
				item_emb=i_emb,
				user_idx=u_idx,
				book_subjects=batch["book_subjects"],
				t=t_contrast,
				use_jaccard=use_jaccard,
				overlap_thresh=overlap_thresh,
			)

			if contrast.numel() == 0:
				total_loss = mse  # fallback if no positives in this batch
				contrast_val = 0.0
			else:
				total_loss = lambda_contrast * contrast + lambda_mse * mse
				contrast_val = contrast.item()

				# log diagnostics
				pos_count_pct += stats["pct_with_positive"].item()
				# stats are on the scaled sim (=cosine/t); we can still compare means across epochs
				if not torch.isnan(stats["pos_mean"]):
					pos_mean_accum += stats["pos_mean"].item()
					neg_mean_accum += stats["neg_mean"].item()
					stat_steps += 1

			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			opt.step()

			# running sums
			sum_total += total_loss.item()
			sum_contrast += contrast_val
			sum_mse += mse.item()

		# epoch summaries
		epoch_rmse = math.sqrt(sse / max(n_examples, 1))
		avg_total = sum_total / step
		avg_contrast = sum_contrast / step
		avg_mse = sum_mse / step

		if stat_steps > 0:
			pct_with_pos = pos_count_pct / step
			pos_mean = pos_mean_accum / stat_steps
			neg_mean = neg_mean_accum / stat_steps
		else:
			pct_with_pos, pos_mean, neg_mean = 0.0, float("nan"), float("nan")

		print(
			f"epoch {epoch+1}/{epochs}  "
			f"total={avg_total:.4f}  "
			f"contrast={avg_contrast:.4f}  "
			f"mse={avg_mse:.4f}  "
			f"rmse={epoch_rmse:.4f}  "
			f"| anchors_with_pos={pct_with_pos:.1f}%  "
			f"pos_mean={pos_mean:.3f}  neg_mean={neg_mean:.3f}"
		)
		state = {
        "subject_embs": model.shared_subj_emb.state_dict(),
        "attn_weight": model.attn_weight.detach().cpu(),
        "attn_bias": model.attn_bias.detach().cpu(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, OUT_PATH)
    print(f"✅ Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()

