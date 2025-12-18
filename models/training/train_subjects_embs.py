import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import math
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

# Shared loaders + attention poolers
from models.training.data_loader import load_rows_and_dataset
from models.training.train_subject_attention import (
    build_pooler_from_env,
    save_components,
)
from models.core import PATHS

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = REPO_ROOT / "models" / "data"

# Map kind → historical filename used by existing loaders
OUT_NAME_BY_KIND = {
    "scalar": "subject_attention_components.pth",
    "perdim": "subject_attention_components_perdim.pth",
    "selfattn": "subject_attention_components_selfattn.pth",
    "selfattn_perdim": "subject_attention_components_selfattn_perdim.pth",
}


def main():
    # -----------------------------
    # Load rows + dataset (supervised)
    # -----------------------------
    rows, ds, n_users, n_items, n_subjects = load_rows_and_dataset(mode="supervised", pad_to=5)
    if not rows or ds is None:
        print("❌ No valid training data found.")
        return

    print(f"✅ Loaded {len(rows)} training samples")

    # Split train/valid like original scripts
    cut = int(0.9 * len(ds))
    train_ds, valid_ds = random_split(ds, [cut, len(ds) - cut])
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=512, device=device)

    # -----------------------------
    # Build model from .env (keeps logic identical to old poolers)
    # -----------------------------
    pooler, kind = build_pooler_from_env(n_users=n_users, n_items=n_items, n_subjects=n_subjects)
    pooler = pooler.to(device)

    # -----------------------------
    # fastai Learner (same loss/metrics/opt as before)
    # -----------------------------
    learn = Learner(
        dls,
        pooler,
        loss_func=MSELossFlat(),
        metrics=[rmse, mae],
        wd=0.05,
        opt_func=Adam,
    )

    from fastai.callback.progress import ProgressCallback

    learn.remove_cbs(ProgressCallback)

    print("🚀 Starting training...")
    # keep schedule consistent with your earlier scripts
    # (you can override with EPOCHS env if desired)
    epochs = int(os.getenv("SUBJ_EPOCHS", "5"))
    lr = float(os.getenv("SUBJ_LR", "3e-2"))
    learn.fit_one_cycle(epochs, lr_max=lr)

    # -----------------------------
    # Save exactly the keys your inference loaders expect
    # -----------------------------
    out_path = PATHS.get_attention_path(kind)
    save_components(pooler, str(out_path), kind)
    print(f"✅ Saved to {out_path}")


if __name__ == "__main__":
    main()
