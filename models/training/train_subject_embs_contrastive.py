import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

# Shared loaders + attention poolers
from models.training.data_loader import load_rows_and_dataset
from models.training.train_subject_attention import (
    build_pooler_from_env,
    save_components,
)
from models.shared_utils import PAD_IDX

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "models" / "data"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Filename map to stay compatible with existing inference loaders
OUT_NAME_BY_KIND = {
    "scalar": "subject_attention_components.pth",
    "perdim": "subject_attention_components_perdim.pth",
    "selfattn": "subject_attention_components_selfattn.pth",
    "selfattn_perdim": "subject_attention_components_selfattn_perdim.pth",
}


# ---------------------------------------------------------
# Contrastive utilities
# ---------------------------------------------------------
@torch.no_grad()
def _jaccard_pos_mask_from_indices(book_subjects: torch.Tensor, thresh_overlap: int = 2) -> torch.Tensor:
    """
    book_subjects: LongTensor [B, L] (padded with PAD_IDX)
    Returns BoolTensor [B, B] where mask[i,j]=True if |S_i ∩ S_j| >= thresh (and i!=j)
    """
    B, L = book_subjects.shape
    sets = []
    for b in range(B):
        row = book_subjects[b].tolist()
        s = set(x for x in row if x != PAD_IDX)
        sets.append(s)
    M = torch.zeros(B, B, dtype=torch.bool, device=book_subjects.device)
    for i in range(B):
        si = sets[i]
        for j in range(i + 1, B):
            if len(si & sets[j]) >= thresh_overlap:
                M[i, j] = True
                M[j, i] = True
    return M


def multi_positive_infonce(item_emb: torch.Tensor,
                           user_idx: torch.Tensor,
                           book_subjects: torch.Tensor,
                           t: float = 0.07,
                           use_jaccard: bool = True,
                           overlap_thresh: int = 2):
    """
    item_emb: [B, D] (not normalized)
    user_idx: [B]
    book_subjects: [B, L]
    Returns: (loss, stats_dict)
    """
    B = item_emb.size(0)
    Z = F.normalize(item_emb, dim=1)
    sim = (Z @ Z.t()) / t                          # [B,B]
    # mask self in denom by a huge negative
    sim = sim - torch.eye(B, device=sim.device) * 1e9

    same_user = user_idx.unsqueeze(1).eq(user_idx.unsqueeze(0))  # [B,B]
    same_user.fill_diagonal_(False)

    if use_jaccard:
        with torch.no_grad():
            subj_mask = _jaccard_pos_mask_from_indices(book_subjects, thresh_overlap=overlap_thresh)
    else:
        subj_mask = torch.zeros_like(same_user)

    pos_mask = (same_user | subj_mask)
    valid = pos_mask.any(dim=1)

    if not valid.any():
        return torch.zeros([], device=item_emb.device), {
            "pct_with_positive": torch.tensor(0.0, device=item_emb.device),
            "pos_mean": torch.tensor(float("nan"), device=item_emb.device),
            "neg_mean": torch.tensor(float("nan"), device=item_emb.device),
        }

    # log-sum-exp numerator over positives, denominator over all
    num = torch.logsumexp(sim.masked_fill(~pos_mask, float("-inf")), dim=1)
    den = torch.logsumexp(sim, dim=1)
    loss_vec = -(num - den)[valid]
    loss = loss_vec.mean()

    with torch.no_grad():
        pos_vals = sim[pos_mask]
        neg_vals = sim[(~pos_mask) & (~torch.eye(B, device=sim.device, dtype=torch.bool))]
        stats = {
            "pct_with_positive": valid.float().mean() * 100.0,
            "pos_mean": pos_vals.mean() if pos_vals.numel() else torch.tensor(float("nan"), device=sim.device),
            "neg_mean": neg_vals.mean() if neg_vals.numel() else torch.tensor(float("nan"), device=sim.device),
        }
    return loss, stats


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def main():
    # Data
    rows, ds, n_users, n_items, n_subjects = load_rows_and_dataset(mode="contrastive", pad_to=5)
    if not rows or ds is None:
        print("❌ No valid training data found.")
        return

    print(f"✅ Loaded {len(rows)} training samples")

    # DataLoader — RandomSampler with replacement improves chance of repeated users per batch
    bs = int(os.getenv("SUBJ_BS", "512"))
    sampler = RandomSampler(ds, replacement=True, num_samples=len(ds))
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    # Model
    pooler, kind = build_pooler_from_env(n_users=n_users, n_items=n_items, n_subjects=n_subjects)
    pooler = pooler.to(device)

    opt = torch.optim.AdamW(pooler.parameters(), lr=float(os.getenv("SUBJ_LR", "3e-3")), weight_decay=5e-2)
    epochs = int(os.getenv("SUBJ_EPOCHS", "6"))

    # Loss weights & params
    lambda_contrast = float(os.getenv("LAMBDA_CONTRAST", "0.8"))
    lambda_mse = float(os.getenv("LAMBDA_MSE", "0.2"))
    t_contrast = float(os.getenv("CONTRAST_T", "0.07"))
    use_jaccard = os.getenv("CONTRAST_USE_JACCARD", "1") != "0"
    overlap_thresh = int(os.getenv("CONTRAST_OVERLAP_THRESH", "2"))

    mse_loss = torch.nn.MSELoss()

    print("🚀 Starting contrastive+RMSE training...")
    for epoch in range(epochs):
        pooler.train()
        sum_total, sum_contrast, sum_mse = 0.0, 0.0, 0.0
        sse, n_examples = 0.0, 0
        pos_count_pct, pos_mean_accum, neg_mean_accum, stat_steps = 0.0, 0.0, 0.0, 0

        for step, batch in enumerate(dl, 1):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # forward through attention pooler
            u_emb = pooler.attention_pool(batch["fav_subjects"])   # [B, D]
            i_emb = pooler.attention_pool(batch["book_subjects"])  # [B, D]
            pred = pooler.rating_head(u_emb, i_emb, batch["user_idx"], batch["item_idx"])  # supervised head

            # --- supervised (MSE) ---
            mse = mse_loss(pred, batch["rating"])  # unweighted MSE

            # accumulate RMSE components
            with torch.no_grad():
                se = (pred.detach() - batch["rating"]) ** 2
                sse += se.sum().item()
                n_examples += se.numel()

            # --- contrastive (multi-positive InfoNCE on items) ---
            contrast, stats = multi_positive_infonce(
                item_emb=i_emb,
                user_idx=batch["user_idx"],
                book_subjects=batch["book_subjects"],
                t=t_contrast,
                use_jaccard=use_jaccard,
                overlap_thresh=overlap_thresh,
            )

            if contrast.numel() == 0:  # no positives in this batch
                total_loss = mse
                contrast_val = 0.0
            else:
                total_loss = lambda_contrast * contrast + lambda_mse * mse
                contrast_val = contrast.item()

                # log diagnostics
                pos_count_pct += stats["pct_with_positive"].item()
                if not torch.isnan(stats["pos_mean"]):
                    pos_mean_accum += stats["pos_mean"].item()
                    neg_mean_accum += stats["neg_mean"].item()
                    stat_steps += 1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pooler.parameters(), max_norm=1.0)
            opt.step()

            sum_total += total_loss.item()
            sum_contrast += contrast_val
            sum_mse += mse.item()

        # epoch metrics
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

    # Save components exactly as inference expects for the chosen kind
    out_name = OUT_NAME_BY_KIND.get(kind, f"subject_attention_components_{kind}.pth")
    out_path = OUT_DIR / out_name
    save_components(pooler, str(out_path), kind)
    print(f"✅ Saved to {out_path}")


if __name__ == "__main__":
    main()
