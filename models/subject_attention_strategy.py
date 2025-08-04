import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shared_utils import PAD_IDX

# ----------------------------
# Strategy registry interface
# ----------------------------
STRATEGY_REGISTRY = {}

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

@register_attention_strategy("scalar")
class ScalarAttentionStrategy(AttentionPoolingStrategy):
    def __init__(self, path: str):
        super().__init__()
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
        device = self.subject_emb.weight.device
        batch_size = len(indices_list)
        max_len = max((len(lst) for lst in indices_list), default=1)

        # Pad input
        padded = [
            lst + [PAD_IDX] * (max_len - len(lst)) if lst else [PAD_IDX] * max_len
            for lst in indices_list
        ]
        idx_tensor = torch.tensor(padded, device=device)
        mask = (idx_tensor != PAD_IDX)

        # Ensure at least one unmasked subject per example
        has_real = mask.any(dim=1)
        for i in range(batch_size):
            if not has_real[i]:
                mask[i, 0] = True

        # Embed and apply attention
        embs = self.subject_emb(idx_tensor)                             # [B, L, D]
        scores = (embs @ self.attn_weight.T) + self.attn_bias          # [B, L, 1]
        scores = scores.squeeze(-1).masked_fill(~mask, float("-inf"))  # [B, L]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)                 # [B, L, 1]
        pooled = (attn * embs).sum(dim=1)                              # [B, D]
        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim
