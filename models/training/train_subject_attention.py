import os
import torch
import torch.nn as nn
from typing import Tuple, Literal

# Import the shared PAD index used across the repo
from models.core import PAD_IDX
from models.core import PATHS

PoolerKind = Literal["scalar", "perdim", "selfattn", "selfattn_perdim"]


# -----------------------------------------------------------------------------
# BasePooler: common bias head + helpers
# -----------------------------------------------------------------------------
class BasePooler(nn.Module):
    """
    Provides the rating head (dot + user/item/global bias) and a common Dropout.
    Concrete subclasses must implement:
      - attention_pool(indices: LongTensor[B,L]) -> FloatTensor[B,D]
      - forward(batch) -> pred (for supervised RMSE trainers)
    """

    def __init__(
        self, n_users: int, n_items: int, n_subjects: int, emb_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

    def rating_head(
        self, u_emb: torch.Tensor, i_emb: torch.Tensor, u_idx: torch.Tensor, i_idx: torch.Tensor
    ) -> torch.Tensor:
        dot = (u_emb * i_emb).sum(dim=1)
        return (
            dot
            + self.user_bias(u_idx).squeeze()
            + self.item_bias(i_idx).squeeze()
            + self.global_bias
        )

    def forward(self, batch) -> torch.Tensor:
        """Default forward used by RMSE trainers.
        Subclasses should not override unless they truly change logic."""
        u = batch["user_idx"]
        i = batch["item_idx"]
        u_sub = batch["fav_subjects"]
        i_sub = batch["book_subjects"]
        u_emb = self.attention_pool(u_sub)
        i_emb = self.attention_pool(i_sub)
        return self.rating_head(u_emb, i_emb, u, i)


# -----------------------------------------------------------------------------
# Scalar attention
# -----------------------------------------------------------------------------
class ScalarPooler(BasePooler):
    def __init__(
        self, n_users: int, n_items: int, n_subjects: int, emb_dim: int = 16, dropout: float = 0.3
    ):
        super().__init__(n_users, n_items, n_subjects, emb_dim, dropout)
        self.shared_subj_emb = nn.Embedding(n_subjects, emb_dim, padding_idx=PAD_IDX)
        self.subject_attn = nn.Linear(emb_dim, 1)

    def attention_pool(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B, L]
        embs = self.shared_subj_emb(indices)  # [B, L, D]
        scores = self.subject_attn(embs).squeeze(-1)  # [B, L]

        mask = indices != PAD_IDX  # [B, L]
        has_real_subjects = mask.any(dim=1)  # [B]

        # Create safe_mask that un-masks one PAD if all are PADs (logic preserved)
        safe_mask = mask.clone()
        for i in range(len(safe_mask)):
            if not has_real_subjects[i]:
                safe_mask[i, 0] = True

        scores = scores.masked_fill(~safe_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # [B, L]
        pooled = (embs * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return self.drop(pooled)


# -----------------------------------------------------------------------------
# Per-dimension attention
# -----------------------------------------------------------------------------
class PerDimPooler(BasePooler):
    def __init__(
        self, n_users: int, n_items: int, n_subjects: int, emb_dim: int = 16, dropout: float = 0.3
    ):
        super().__init__(n_users, n_items, n_subjects, emb_dim, dropout)
        self.shared_subj_emb = nn.Embedding(n_subjects, emb_dim, padding_idx=PAD_IDX)
        self.attn_weight = nn.Parameter(torch.empty(emb_dim))
        self.attn_bias = nn.Parameter(torch.empty(emb_dim))
        nn.init.xavier_uniform_(self.attn_weight.unsqueeze(0))
        nn.init.zeros_(self.attn_bias)

    def attention_pool(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B, L]
        embs = self.shared_subj_emb(indices)  # [B, L, D]
        scores = (embs * self.attn_weight) + self.attn_bias  # [B, L, D]
        scores = scores.sum(dim=-1)  # [B, L]

        mask = indices != PAD_IDX  # [B, L]
        has_real = mask.any(dim=1)  # [B]
        # Preserve original safety logic (flip first token to True if all PAD)
        for b in range(len(mask)):
            if not has_real[b]:
                mask[b, 0] = True

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B, L]
        pooled = (embs * attn.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return self.drop(pooled)


# -----------------------------------------------------------------------------
# Self-attention block + CLS pooling
# -----------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        attn_drop: float = 0.1,
        ffn_mult: int = 2,
        ffn_drop: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_drop),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask
        )
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class SelfAttentionPooler(BasePooler):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_subjects: int,
        emb_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(n_users, n_items, n_subjects, emb_dim, dropout)
        self.subject_emb = nn.Embedding(n_subjects, emb_dim, padding_idx=PAD_IDX)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.block = SelfAttentionBlock(
            d_model=emb_dim, n_heads=n_heads, attn_drop=dropout, ffn_drop=dropout
        )

    def attention_pool(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B, L]
        mask = indices != PAD_IDX  # [B, L]
        embs = self.subject_emb(indices)  # [B, L, D]
        B = indices.size(0)
        cls = self.cls_token.expand(B, 1, -1)  # [B, 1, D]
        x = torch.cat([cls, embs], dim=1)  # [B, 1+L, D]
        mask_ext = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        x = self.block(x, key_padding_mask=~mask_ext)
        pooled = x[:, 0, :]  # CLS pooling
        return self.drop(pooled)


# -----------------------------------------------------------------------------
# Self-attention + Per-dim attention head
# -----------------------------------------------------------------------------
class SelfAttnPerDimPooler(BasePooler):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_subjects: int,
        emb_dim: int = 16,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(n_users, n_items, n_subjects, emb_dim, dropout)
        self.subject_emb = nn.Embedding(n_subjects, emb_dim, padding_idx=PAD_IDX)
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.attn_weight = nn.Parameter(torch.empty(emb_dim))
        self.attn_bias = nn.Parameter(torch.empty(emb_dim))
        nn.init.xavier_uniform_(self.attn_weight.unsqueeze(0))
        nn.init.zeros_(self.attn_bias)

    def attention_pool(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B, L]
        mask = indices != PAD_IDX
        embs = self.subject_emb(indices)  # [B, L, D]
        attn_out, _ = self.mha(embs, embs, embs, key_padding_mask=~mask)  # [B, L, D]

        # Per-dim attention head
        scores = (attn_out * self.attn_weight) + self.attn_bias  # [B, L, D]
        scores = scores.sum(dim=-1).masked_fill(~mask, float("-inf"))  # [B, L]
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = (attn * attn_out).sum(dim=1)  # [B, D]
        return self.drop(pooled)


# -----------------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------------
_DEF_KIND: PoolerKind = "perdim"


def build_pooler(
    kind: PoolerKind = None,
    n_users: int = 1,
    n_items: int = 1,
    n_subjects: int = 1,
    emb_dim: int = 16,
    dropout: float = 0.1,
    n_heads: int = 4,
) -> BasePooler:
    """Instantiate a pooler by kind."""
    k = (kind or _DEF_KIND).lower()
    if k == "scalar":
        return ScalarPooler(n_users, n_items, n_subjects, emb_dim=emb_dim, dropout=dropout)
    if k == "perdim":
        return PerDimPooler(n_users, n_items, n_subjects, emb_dim=emb_dim, dropout=dropout)
    if k == "selfattn":
        # default emb_dim=64 for self-attn per your current trainer
        return SelfAttentionPooler(
            n_users, n_items, n_subjects, emb_dim=emb_dim, n_heads=n_heads, dropout=dropout
        )
    if k == "selfattn_perdim":
        return SelfAttnPerDimPooler(
            n_users, n_items, n_subjects, emb_dim=emb_dim, n_heads=n_heads, dropout=dropout
        )
    raise ValueError(f"Unknown attention kind: {kind}")


def build_pooler_from_env(
    n_users: int,
    n_items: int,
    n_subjects: int,
    emb_dim: int = None,
    dropout: float = None,
    n_heads: int = None,
) -> Tuple[BasePooler, PoolerKind]:
    """
    Build via env variables:
      SUBJ_ATTENTION   = scalar | perdim | selfattn | selfattn_perdim (default: perdim)
      SUBJ_EMB_DIM     = int (default aligns with each class' current usage)
      SUBJ_DROPOUT     = float (default aligns with trainers)
      SUBJ_N_HEADS     = int (for self-attn variants; default 4)
    """
    kind: PoolerKind = os.getenv("SUBJ_ATTENTION", _DEF_KIND)  # type: ignore
    # sensible defaults matching your existing scripts
    if emb_dim is None:
        emb_dim = int(os.getenv("SUBJ_EMB_DIM", 64))
    if dropout is None:
        dropout = float(os.getenv("SUBJ_DROPOUT", 0.1 if kind.startswith("selfattn") else 0.3))
    if n_heads is None:
        n_heads = int(os.getenv("SUBJ_N_HEADS", 4))

    pooler = build_pooler(
        kind,
        n_users=n_users,
        n_items=n_items,
        n_subjects=n_subjects,
        emb_dim=emb_dim,
        dropout=dropout,
        n_heads=n_heads,
    )
    return pooler, kind  # return kind so caller can branch on save-keys if needed


# -----------------------------------------------------------------------------
# Save helpers to keep file formats identical to current loaders
# -----------------------------------------------------------------------------
@torch.no_grad()
def save_components(pooler: BasePooler, out_path: str, kind: PoolerKind):
    state = {}
    if kind == "scalar":
        m: ScalarPooler = pooler  # type: ignore
        state = {
            "subject_embs": m.shared_subj_emb.state_dict(),
            "attn_weight": m.subject_attn.weight.detach().cpu(),
            "attn_bias": m.subject_attn.bias.detach().cpu(),
        }
    elif kind == "perdim":
        m: PerDimPooler = pooler  # type: ignore
        state = {
            "subject_embs": m.shared_subj_emb.state_dict(),
            "attn_weight": m.attn_weight.detach().cpu(),
            "attn_bias": m.attn_bias.detach().cpu(),
        }
    elif kind == "selfattn":
        m: SelfAttentionPooler = pooler  # type: ignore
        state = {
            "subject_embs": m.subject_emb.state_dict(),
            "sab": m.block.state_dict(),
            "cls_token": m.cls_token.detach().cpu(),
            "n_heads": int(m.block.attn.num_heads),
        }
    elif kind == "selfattn_perdim":
        m: SelfAttnPerDimPooler = pooler  # type: ignore
        state = {
            "subject_embs": m.subject_emb.state_dict(),
            "mha": m.mha.state_dict(),
            "attn_weight": m.attn_weight.detach().cpu(),
            "attn_bias": m.attn_bias.detach().cpu(),
        }
    else:
        raise ValueError(f"Unknown attention kind for saving: {kind}")

    PATHS.ensure_staging_dirs()
    torch.save(state, out_path)
