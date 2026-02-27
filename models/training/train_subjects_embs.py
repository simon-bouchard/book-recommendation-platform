# models/training/train_subjects_embs.py
"""
Supervised subject embedding training with configurable PAD_IDX.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import random_split
from fastai.learner import Learner
from fastai.metrics import rmse, mae
from fastai.data.core import DataLoaders
from fastai.losses import MSELossFlat
from fastai.optimizer import Adam
from fastprogress.fastprogress import progress_bar

progress_bar.NO_BAR = True

device = "cuda" if torch.cuda.is_available() else "cpu"

from models.training.data_loader import load_rows_and_dataset
from models.training.train_subject_attention import (
    build_pooler_from_env,
    save_components,
)
from models.training.metrics import record_training_metrics

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "models" / "data"

OUT_NAME_BY_KIND = {
    "scalar": "subject_attention_components.pth",
    "perdim": "subject_attention_components_perdim.pth",
    "selfattn": "subject_attention_components_selfattn.pth",
    "selfattn_perdim": "subject_attention_components_selfattn_perdim.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train subject embeddings with supervised RMSE loss"
    )
    parser.add_argument(
        "--pad-idx",
        type=int,
        default=None,
        help="Padding index for invalid subjects (default: from PAD_IDX env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use provided PAD_IDX or fall back to environment variable
    pad_idx = args.pad_idx if args.pad_idx is not None else int(os.getenv("PAD_IDX", "0"))

    print(f"Using PAD_IDX = {pad_idx}")

    # Load rows + dataset with custom PAD_IDX
    rows, ds, n_users, n_items, n_subjects = load_rows_and_dataset(
        mode="supervised", pad_to=5, pad_idx=pad_idx
    )

    if not rows or ds is None:
        print("No valid training data found.")
        return

    print(f"Loaded {len(rows)} training samples")

    # Split train/valid
    cut = int(0.9 * len(ds))
    train_ds, valid_ds = random_split(ds, [cut, len(ds) - cut])
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=512, device=device)

    # Build model (pass n_subjects which includes PAD_IDX)
    pooler, kind = build_pooler_from_env(n_users=n_users, n_items=n_items, n_subjects=n_subjects)
    pooler = pooler.to(device)

    # Update pooler to use correct PAD_IDX
    if hasattr(pooler, "shared_subj_emb"):
        pooler.shared_subj_emb.padding_idx = pad_idx
    elif hasattr(pooler, "subject_emb"):
        pooler.subject_emb.padding_idx = pad_idx

    # Create learner
    learn = Learner(
        dls, pooler, loss_func=MSELossFlat(), metrics=[rmse, mae], wd=0.05, opt_func=Adam
    )

    from fastai.callback.progress import ProgressCallback

    learn.remove_cbs(ProgressCallback)

    print("Starting training...")
    epochs = int(os.getenv("SUBJ_EPOCHS", "5"))
    lr = float(os.getenv("SUBJ_LR", "3e-2"))
    learn.fit_one_cycle(epochs, lr_max=lr)

    record_training_metrics(
        "subject_embeddings",
        {
            "final_rmse": float(learn.recorder.values[-1][1]),
            "final_mae": float(learn.recorder.values[-1][2]),
            "epochs": epochs,
            "kind": kind,
            "n_train_samples": len(train_ds),
        },
    )
    print("Recorded training metrics")

    # Save
    out_name = OUT_NAME_BY_KIND.get(kind, f"subject_attention_components_{kind}.pth")
    out_path = OUT_DIR / out_name
    save_components(pooler, str(out_path), kind)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
