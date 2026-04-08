import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.constants import PAD_IDX

# ----------------------------
# Strategy registry interface
# ----------------------------
STRATEGY_REGISTRY = {}


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_drop: float = 0.0,
        ffn_mult: int = 2,
        ffn_drop: float = 0.0,
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
        # key_padding_mask=True means "ignore" for nn.MultiheadAttention
        y, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask
        )
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


def register_attention_strategy(name):
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


class AttentionPoolingStrategy(nn.Module):
    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        raise NotImplementedError

    def prepare_inputs(self, indices_list: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pads input and returns (idx_tensor, mask)"""
        device = self.subject_emb.weight.device
        batch_size = len(indices_list)
        max_len = max((len(lst) for lst in indices_list), default=1)

        # Pad and create mask
        padded = [
            lst + [PAD_IDX] * (max_len - len(lst)) if lst else [PAD_IDX] * max_len
            for lst in indices_list
        ]
        idx_tensor = torch.tensor(padded, device=device)
        mask = idx_tensor != PAD_IDX

        # Ensure at least one valid entry per row
        has_real = mask.any(dim=1)
        for i in range(batch_size):
            if not has_real[i]:
                mask[i, 0] = True

        return idx_tensor, mask


@register_attention_strategy("scalar")
class ScalarAttentionStrategy(AttentionPoolingStrategy):
    def __init__(self, path: str):
        super().__init__()
        print("Using ScalarAttentionStrategy")
        state = torch.load(path, map_location="cpu")

        # Load frozen subject embedding
        self.subject_emb = nn.Embedding.from_pretrained(
            state["subject_embs"]["weight"],
            freeze=True,
            padding_idx=PAD_IDX,
        )

        # Register attention weights as non-trainable buffers
        self.register_buffer("attn_weight", state["attn_weight"])
        self.register_buffer("attn_bias", state["attn_bias"])

    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        idx_tensor, mask = self.prepare_inputs(indices_list)
        embs = self.subject_emb(idx_tensor)  # [B, L, D]

        scores = (embs @ self.attn_weight.T) + self.attn_bias  # [B, L, 1]
        scores = scores.squeeze(-1).masked_fill(~mask, float("-inf"))  # [B, L]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = (attn * embs).sum(dim=1)  # [B, D]
        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim


@register_attention_strategy("perdim")
class PerDimAttentionStrategy(AttentionPoolingStrategy):
    def __init__(self, path: str):
        super().__init__()
        state = torch.load(path, map_location="cpu")

        self.subject_emb = nn.Embedding.from_pretrained(
            state["subject_embs"]["weight"],
            freeze=True,
            padding_idx=PAD_IDX,
        )

        self.register_buffer("attn_weight", state["attn_weight"])  # shape [D]
        self.register_buffer("attn_bias", state["attn_bias"])  # shape [D]

    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        idx_tensor, mask = self.prepare_inputs(indices_list)
        embs = self.subject_emb(idx_tensor)  # [B, L, D]

        scores = (embs * self.attn_weight) + self.attn_bias  # [B, L, D]
        scores = scores.sum(dim=-1)  # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))  # [B, L]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]

        pooled = (attn * embs).sum(dim=1)  # [B, D]
        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim


@register_attention_strategy("selfattn")
class SelfAttentionStrategy(AttentionPoolingStrategy):
    def __init__(self, path: str):
        super().__init__()
        state = torch.load(path, map_location="cpu")

        # required keys from the new training saver
        for k in ("subject_embs", "sab", "cls_token", "n_heads"):
            if k not in state:
                raise ValueError(f"Missing '{k}' in self-attn state file: {path}")

        # frozen subject embeddings for inference
        self.subject_emb = nn.Embedding.from_pretrained(
            state["subject_embs"]["weight"],
            freeze=True,
            padding_idx=PAD_IDX,
        )

        D = self.subject_emb.embedding_dim
        n_heads = int(state["n_heads"])

        # instantiate SAB and load trained weights
        self.sab = SelfAttentionBlock(d_model=D, n_heads=n_heads, attn_drop=0.0, ffn_drop=0.0)
        self.sab.load_state_dict(state["sab"], strict=True)

        # learned CLS token
        cls = state["cls_token"]
        if cls.shape != (1, 1, D):
            raise ValueError(f"cls_token has shape {cls.shape}, expected (1,1,{D})")
        self.cls_token = nn.Parameter(cls)

        # no dropout at inference
        self.drop = nn.Identity()

        # inference mode
        self.eval()

    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        idx_tensor, mask = self.prepare_inputs(indices_list)  # [B, L], [B, L]=True for valid
        embs = self.subject_emb(idx_tensor)  # [B, L, D]

        B, L, D = embs.shape
        cls = self.cls_token.expand(B, 1, D)  # [B, 1, D]
        x = torch.cat([cls, embs], dim=1)  # [B, 1+L, D]

        # extend mask (CLS is always valid)
        mask_ext = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
        )  # [B,1+L]
        x = self.sab(x, key_padding_mask=~mask_ext)  # MHA expects True=ignore

        pooled = x[:, 0, :]  # CLS-pooled vector [B, D]
        pooled = self.drop(pooled)
        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim


@register_attention_strategy("selfattn_perdim")
class SelfAttnPerDimStrategy(AttentionPoolingStrategy):
    def __init__(self, path: str):
        super().__init__()
        state = torch.load(path, map_location="cpu")

        # Trainable subject embedding
        self.subject_emb = nn.Embedding.from_pretrained(
            embeddings=state["subject_embs"]["weight"],
            freeze=False,
            padding_idx=PAD_IDX,
        )

        D = self.subject_emb.embedding_dim
        self.mha = nn.MultiheadAttention(embed_dim=D, num_heads=2, batch_first=True)
        self.mha.load_state_dict(state["mha"])

        self.register_buffer("attn_weight", state["attn_weight"])  # [D]
        self.register_buffer("attn_bias", state["attn_bias"])  # [D]

    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        idx_tensor, mask = self.prepare_inputs(indices_list)  # [B, L], [B, L]
        embs = self.subject_emb(idx_tensor)  # [B, L, D]

        attn_out, _ = self.mha(embs, embs, embs, key_padding_mask=~mask)  # [B, L, D]

        # Per-dim attention over contextualized output
        scores = (attn_out * self.attn_weight) + self.attn_bias  # [B, L, D]
        scores = scores.sum(dim=-1).masked_fill(~mask, float("-inf"))  # [B, L]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]

        pooled = (attn * attn_out).sum(dim=1)  # [B, D]
        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim
