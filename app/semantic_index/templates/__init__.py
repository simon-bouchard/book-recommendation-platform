# app/semantic_index/templates/__init__.py
"""
Text template functions for embedding construction.
"""

from .text_templates import (
    build_baseline_old_text,
    build_baseline_clean_text,
    build_v1_full_text,
    build_v1_subjects_text,
    build_v2_full_text,
    build_v2_subjects_text,
    validate_text_not_empty,
)

__all__ = [
    "build_baseline_old_text",
    "build_baseline_clean_text",
    "build_v1_full_text",
    "build_v1_subjects_text",
    "build_v2_full_text",
    "build_v2_subjects_text",
    "validate_text_not_empty",
]
