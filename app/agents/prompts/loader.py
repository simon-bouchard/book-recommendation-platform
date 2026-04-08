from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

PROMPTS_DIR_ENV = "PROMPTS_DIR"  # optional override for rapid iteration


def read_prompt(name: str) -> str:
    env_dir = os.getenv(PROMPTS_DIR_ENV)
    if env_dir:
        return (Path(env_dir) / name).read_text(encoding="utf-8")
    return (files(__package__) / name).read_text(encoding="utf-8")
