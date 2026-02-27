# models/training/metrics.py
"""
Utilities for recording and loading training run metrics.

Each training script calls record_training_metrics() at completion.
All scripts in a run write to the same JSON file, which is then SCP'd
back alongside model artifacts for use by the deployment gate.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from models.core import PATHS

_METRICS_FILENAME = "training_metrics.json"


def _default_output_path() -> Path:
    """
    Resolve the metrics output path.

    Checks the TRAINING_METRICS_PATH environment variable first,
    then falls back to PATHS.artifacts_dir. The artifacts_dir location
    ensures the file is included in the standard SCP artifact transfer
    without any changes to automated_training.py.
    """
    env_override = os.getenv("TRAINING_METRICS_PATH")
    if env_override:
        return Path(env_override)
    return PATHS.artifacts_dir / _METRICS_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON to path atomically using a sibling temp file and os.replace.

    Protects against corrupt files if the process is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=".tmp_metrics_",
        suffix=".json",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, path)


def record_training_metrics(
    script: str,
    metrics: Dict[str, Any],
    duration_s: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    Append or update metrics for a single training script into the shared metrics file.

    Reads the existing file (if any), merges the new entry under the script's
    key, and writes atomically. Safe to call from multiple sequential scripts
    since automated_training.py runs them one at a time.

    The metrics dict is intentionally untyped — different scripts record
    different things. GBT scripts record rmse/mae; ALS records training
    statistics like n_warm_users and n_items.

    Args:
        script: Identifying name for the training script, e.g. "warm_gbt",
                "cold_gbt", "als", "subject_embeddings".
        metrics: Arbitrary key/value pairs to record. All values must be
                 JSON-serializable (float, int, str, bool, None).
        duration_s: Wall-clock training duration in seconds, if available.
        output_path: Override the output path. Defaults to
                     PATHS.artifacts_dir / "training_metrics.json".
    """
    path = output_path or _default_output_path()

    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    entry: Dict[str, Any] = {
        "metrics": metrics,
        "recorded_at": _now_iso(),
    }
    if duration_s is not None:
        entry["duration_s"] = round(duration_s, 1)

    scripts = existing.get("scripts", {})
    scripts[script] = entry

    payload: Dict[str, Any] = {
        "last_updated": _now_iso(),
        "scripts": scripts,
    }

    _atomic_write(path, payload)
    print(f"Metrics recorded for '{script}' -> {path}")


def load_training_metrics(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the metrics file produced by a training run.

    Returns the full parsed document. The "scripts" key maps each script
    name to its recorded entry. Returns an empty dict if the file does
    not exist, so callers can treat a missing file the same as an empty run.

    Args:
        path: Path to the metrics JSON file. Defaults to the standard location.

    Returns:
        Parsed metrics document, or an empty dict if the file is absent.

    Raises:
        json.JSONDecodeError: If the file exists but is not valid JSON.
    """
    path = path or _default_output_path()

    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def get_script_metrics(
    script: str,
    path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convenience accessor for a single script's metrics entry.

    Args:
        script: Script name key, e.g. "warm_gbt".
        path: Path to the metrics JSON file. Defaults to the standard location.

    Returns:
        The entry dict for that script, or None if absent.
    """
    doc = load_training_metrics(path)
    return doc.get("scripts", {}).get(script)
