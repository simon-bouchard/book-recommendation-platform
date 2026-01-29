# models/training/train_subject_embs_contrastive.py
"""
Contrastive subject embedding training optimized for CPU parallelism.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from models.training.data_loader import load_rows_and_dataset
from models.training.train_subject_attention import (
    build_pooler_from_env,
    save_components,
)

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "models" / "artifacts" / "attention"

device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_NAME_BY_KIND = {
    "scalar": "subject_attention_components.pth",
    "perdim": "subject_attention_components_perdim.pth",
    "selfattn": "subject_attention_components_selfattn.pth",
    "selfattn_perdim": "subject_attention_components_selfattn_perdim.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train subject embeddings with contrastive + RMSE loss"
    )
    parser.add_argument(
        "--pad-idx",
        type=int,
        default=None,
        help="Padding index for invalid subjects (default: from PAD_IDX env var)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers (default: CPU cores - 2, min 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 1024, increase for better CPU utilization)",
    )
    return parser.parse_args()


@torch.no_grad()
def _jaccard_pos_mask_from_indices(
    book_subjects: torch.Tensor, thresh_overlap: int, pad_idx: int
) -> torch.Tensor:
    """Build positive mask based on subject overlap."""
    B, L = book_subjects.shape
    sets = []
    for b in range(B):
        row = book_subjects[b].tolist()
        s = set(x for x in row if x != pad_idx)
        sets.append(s)

    M = torch.zeros(B, B, dtype=torch.bool, device=book_subjects.device)
    for i in range(B):
        si = sets[i]
        for j in range(i + 1, B):
            if len(si & sets[j]) >= thresh_overlap:
                M[i, j] = True
                M[j, i] = True
    return M


def multi_positive_infonce(
    item_emb: torch.Tensor,
    user_idx: torch.Tensor,
    book_subjects: torch.Tensor,
    pad_idx: int,
    t: float = 0.07,
    use_jaccard: bool = True,
    overlap_thresh: int = 2,
):
    """Multi-positive InfoNCE loss."""
    B = item_emb.size(0)
    Z = F.normalize(item_emb, dim=1)
    sim = (Z @ Z.t()) / t
    sim = sim - torch.eye(B, device=sim.device) * 1e9

    same_user = user_idx.unsqueeze(1).eq(user_idx.unsqueeze(0))
    same_user.fill_diagonal_(False)

    if use_jaccard:
        subj_mask = _jaccard_pos_mask_from_indices(book_subjects, overlap_thresh, pad_idx)
    else:
        subj_mask = torch.zeros_like(same_user)

    pos_mask = same_user | subj_mask
    valid = pos_mask.any(dim=1)

    if not valid.any():
        return torch.zeros([], device=item_emb.device), {
            "pct_with_positive": torch.tensor(0.0, device=item_emb.device),
            "pos_mean": torch.tensor(float("nan"), device=item_emb.device),
            "neg_mean": torch.tensor(float("nan"), device=item_emb.device),
        }

    num = torch.logsumexp(sim.masked_fill(~pos_mask, float("-inf")), dim=1)
    den = torch.logsumexp(sim, dim=1)
    loss_vec = -(num - den)[valid]
    loss = loss_vec.mean()

    with torch.no_grad():
        pos_vals = sim[pos_mask]
        neg_vals = sim[(~pos_mask) & (~torch.eye(B, device=sim.device, dtype=torch.bool))]
        stats = {
            "pct_with_positive": valid.float().mean() * 100.0,
            "pos_mean": pos_vals.mean()
            if pos_vals.numel()
            else torch.tensor(float("nan"), device=sim.device),
            "neg_mean": neg_vals.mean()
            if neg_vals.numel()
            else torch.tensor(float("nan"), device=sim.device),
        }
    return loss, stats


def configure_cpu_parallelism():
    """
    Configure PyTorch for optimal CPU utilization.

    Sets inter-op and intra-op parallelism based on available cores.
    """
    import multiprocessing as mp

    num_cores = mp.cpu_count()

    # Set number of threads for PyTorch operations
    # Use all available cores for intra-op (within operations like matmul)
    torch.set_num_threads(num_cores)

    # Set inter-op threads (between operations) to 1 or 2 to avoid oversubscription
    torch.set_num_interop_threads(min(2, num_cores // 2))

    print(f"CPU Configuration:")
    print(f"  - Available cores: {num_cores}")
    print(f"  - Intra-op threads (computation): {torch.get_num_threads()}")
    print(f"  - Inter-op threads (parallelism): {torch.get_num_interop_threads()}")

    return num_cores


def main():
    args = parse_args()

    # Configure CPU parallelism BEFORE creating model
    num_cores = configure_cpu_parallelism()

    # Use provided PAD_IDX or fall back to environment variable
    pad_idx = args.pad_idx if args.pad_idx is not None else int(os.getenv("PAD_IDX", "0"))

    print(f"Using PAD_IDX = {pad_idx}")
    print(f"Device: {device}")

    # Load data with custom PAD_IDX
    rows, ds, n_users, n_items, n_subjects = load_rows_and_dataset(
        mode="contrastive", pad_to=5, pad_idx=pad_idx
    )

    if not rows or ds is None:
        print("No valid training data found.")
        return

    print(f"Loaded {len(rows)} training samples")

    # Determine optimal batch size and num_workers
    if args.batch_size:
        bs = args.batch_size
    else:
        # Larger batch size for CPU to amortize overhead
        bs = int(os.getenv("SUBJ_BS", "2048"))  # Increased from 1024

    if args.num_workers:
        num_workers = args.num_workers
    else:
        # Use most cores for data loading (leave 2 for computation)
        num_workers = max(4, num_cores - 2)

    print(f"DataLoader Configuration:")
    print(f"  - Batch size: {bs}")
    print(f"  - Num workers: {num_workers}")

    sampler = RandomSampler(ds, replacement=True, num_samples=len(ds))

    # Only use pin_memory if CUDA is available
    use_pin_memory = torch.cuda.is_available()

    dl = DataLoader(
        ds,
        batch_size=bs,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches per worker
    )

    # Model
    pooler, kind = build_pooler_from_env(n_users=n_users, n_items=n_items, n_subjects=n_subjects)
    pooler = pooler.to(device)

    # Update pooler to use correct PAD_IDX
    if hasattr(pooler, "shared_subj_emb"):
        pooler.shared_subj_emb.padding_idx = pad_idx
    elif hasattr(pooler, "subject_emb"):
        pooler.subject_emb.padding_idx = pad_idx

    opt = torch.optim.AdamW(
        pooler.parameters(), lr=float(os.getenv("SUBJ_LR", "3e-3")), weight_decay=5e-2
    )
    epochs = int(os.getenv("SUBJ_EPOCHS", "14"))

    lambda_contrast = float(os.getenv("LAMBDA_CONTRAST", "0.8"))
    lambda_mse = float(os.getenv("LAMBDA_MSE", "0.2"))
    t_contrast = float(os.getenv("CONTRAST_T", "0.07"))
    use_jaccard = os.getenv("CONTRAST_USE_JACCARD", "1") != "0"
    overlap_thresh = int(os.getenv("CONTRAST_OVERLAP_THRESH", "2"))

    mse_loss = torch.nn.MSELoss()

    print("Starting contrastive+RMSE training...")

    import time

    for epoch in range(epochs):
        pooler.train()
        sum_total, sum_contrast, sum_mse = 0.0, 0.0, 0.0
        sse, n_examples = 0.0, 0
        pos_count_pct, pos_mean_accum, neg_mean_accum, stat_steps = 0.0, 0.0, 0.0, 0

        epoch_start = time.time()

        for step, batch in enumerate(dl, 1):
            # Use non_blocking only if CUDA is available
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=use_pin_memory)

            opt.zero_grad(set_to_none=True)

            u_emb = pooler.attention_pool(batch["fav_subjects"])
            i_emb = pooler.attention_pool(batch["book_subjects"])
            pred = pooler.rating_head(u_emb, i_emb, batch["user_idx"], batch["item_idx"])

            mse = mse_loss(pred, batch["rating"])

            with torch.no_grad():
                se = (pred.detach() - batch["rating"]) ** 2
                sse += se.sum().item()
                n_examples += se.numel()

            contrast, stats = multi_positive_infonce(
                item_emb=i_emb,
                user_idx=batch["user_idx"],
                book_subjects=batch["book_subjects"],
                pad_idx=pad_idx,
                t=t_contrast,
                use_jaccard=use_jaccard,
                overlap_thresh=overlap_thresh,
            )

            if contrast.numel() == 0:
                total_loss = mse
                contrast_val = 0.0
            else:
                total_loss = lambda_contrast * contrast + lambda_mse * mse
                contrast_val = contrast.item()

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

        epoch_time = time.time() - epoch_start
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
            f"epoch {epoch + 1}/{epochs}  "
            f"time={epoch_time:.1f}s  "
            f"total={avg_total:.4f}  "
            f"contrast={avg_contrast:.4f}  "
            f"mse={avg_mse:.4f}  "
            f"rmse={epoch_rmse:.4f}  "
            f"| anchors_with_pos={pct_with_pos:.1f}%  "
            f"pos_mean={pos_mean:.3f}  neg_mean={neg_mean:.3f}"
        )

    # Save
    out_name = OUT_NAME_BY_KIND.get(kind, f"subject_attention_components_{kind}.pth")
    out_path = OUT_DIR / out_name
    save_components(pooler, str(out_path), kind)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
