# models/core/config.py
"""
Configuration management for models module, primarily handling environment variables.
"""

import os
from typing import Literal

AttentionStrategy = Literal["scalar", "perdim", "selfattn", "selfattn_perdim"]


class Config:
    """
    Centralized configuration for the models package.

    All configuration values are loaded from environment variables with sensible defaults.
    """

    @staticmethod
    def get_attention_strategy() -> AttentionStrategy:
        """Get the attention pooling strategy from environment."""
        strategy = os.getenv("ATTN_STRATEGY", "scalar").lower()
        valid_strategies = {"scalar", "perdim", "selfattn", "selfattn_perdim"}

        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid ATTN_STRATEGY: {strategy}. Must be one of: {', '.join(valid_strategies)}"
            )

        return strategy  # type: ignore

    @staticmethod
    def get_training_config() -> dict:
        """Get training-related configuration from environment."""
        return {
            "subject_emb_dim": int(os.getenv("SUBJ_EMB_DIM", "64")),
            "subject_dropout": float(os.getenv("SUBJ_DROPOUT", "0.3")),
            "subject_n_heads": int(os.getenv("SUBJ_N_HEADS", "4")),
            "subject_batch_size": int(os.getenv("SUBJ_BS", "1024")),
            "subject_learning_rate": float(os.getenv("SUBJ_LR", "3e-3")),
            "subject_epochs": int(os.getenv("SUBJ_EPOCHS", "14")),
            "subject_auto_train": os.getenv("SUBJECT_AUTO_TRAIN", "false").lower() == "true",
        }

    @staticmethod
    def get_contrastive_config() -> dict:
        """Get contrastive learning configuration from environment."""
        return {
            "lambda_contrast": float(os.getenv("LAMBDA_CONTRAST", "0.8")),
            "lambda_mse": float(os.getenv("LAMBDA_MSE", "0.2")),
            "contrast_temperature": float(os.getenv("CONTRAST_T", "0.07")),
            "use_jaccard": os.getenv("CONTRAST_USE_JACCARD", "1") != "0",
            "overlap_threshold": int(os.getenv("CONTRAST_OVERLAP_THRESH", "2")),
        }
