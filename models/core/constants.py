# models/core/constants.py
"""
System-wide constants used throughout the models package.
"""

import os

PAD_IDX: int = int(os.getenv("PAD_IDX", "0"))
