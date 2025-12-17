# models/core/constants.py
"""
System-wide constants used throughout the models package.
"""

import os
import torch

PAD_IDX: int = int(os.getenv("PAD_IDX", "0"))

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
