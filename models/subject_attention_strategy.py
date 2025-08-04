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
        mask = (idx_tensor != PAD_IDX)

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
        embs = self.subject_emb(idx_tensor)                             # [B, L, D]
        
        scores = (embs @ self.attn_weight.T) + self.attn_bias          # [B, L, 1]
        scores = scores.squeeze(-1).masked_fill(~mask, float("-inf"))  # [B, L]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)                 # [B, L, 1]
        pooled = (attn * embs).sum(dim=1)                              # [B, D]
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
        self.register_buffer("attn_bias", state["attn_bias"])      # shape [D]

    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        idx_tensor, mask = self.prepare_inputs(indices_list)
        embs = self.subject_emb(idx_tensor)  # [B, L, D]

        scores = (embs * self.attn_weight) + self.attn_bias  # [B, L, D]
        scores = scores.sum(dim=-1)                          # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))    # [B, L]
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)       # [B, L, 1]

        pooled = (attn * embs).sum(dim=1)                    # [B, D]
        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim

@register_attention_strategy("selfattn")
class SelfAttentionStrategy(AttentionPoolingStrategy):
    def __init__(self, path: str):
        super().__init__()
        state = torch.load(path, map_location="cpu")

        # Trainable subject embedding
        self.subject_emb = nn.Embedding.from_pretrained(
            embeddings=state["subject_embs"]["weight"],
            freeze=False,  # trainable
            padding_idx=PAD_IDX,
        )

        D = self.subject_emb.embedding_dim
        self.mha = nn.MultiheadAttention(embed_dim=D, num_heads=2, batch_first=True)
        self.mha.load_state_dict(state["mha"])

    def forward(self, indices_list: list[list[int]]) -> torch.Tensor:
        idx_tensor, mask = self.prepare_inputs(indices_list)        # [B, L], [B, L]
        embs = self.subject_emb(idx_tensor)                         # [B, L, D]

        # MHA expects True where PAD — opposite of our mask
        attn_output, _ = self.mha(
            embs, embs, embs,
            key_padding_mask=(~mask)  # [B, L]
        )  # attn_output: [B, L, D]

        # Mean-pooling over valid positions
        masked_output = attn_output.masked_fill(~mask.unsqueeze(-1), 0.0)  # [B, L, D]
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)              # [B, 1]
        pooled = masked_output.sum(dim=1) / lengths                       # [B, D]

        return pooled.nan_to_num(0.0)

    def get_embedding_dim(self) -> int:
        return self.subject_emb.embedding_dim
