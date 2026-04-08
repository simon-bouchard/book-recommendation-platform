# ops/training/automated_training.py
"""
Automated training pipeline that orchestrates local training, artifact staging,
quality gate evaluation, versioned promotion, and worker reload signaling.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from models.core.artifact_registry import (
    generate_version_id,
    promote_staging,
    retire_old_versions,
)
from models.core.paths import PATHS
from ops.training.evaluate_gate import evaluate
from ops.training.notify import notify, read_tail as _notify_read_tail
from ops.training.reload_signal import signal_workers_reload

# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------

LOCAL_PAD_IDX = int(os.getenv("PAD_IDX", "0"))
LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR", str(PROJECT_ROOT / "models/training/logs")))

# ---------------------------------------------------------------------------
# Training script selection
# ---------------------------------------------------------------------------

ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar").lower()
SUBJECT_AUTO_TRAIN = os.getenv("SUBJECT_AUTO_TRAIN", "false").lower() == "true"
_subject_script = os.getenv("SUBJECT_TRAIN_FILE", "train_subject_embs_scalar.py")

if not (PROJECT_ROOT / "models/training" / _subject_script).exists():
    _subject_script = "train_subject_embs_scalar.py"

TRAIN_SCRIPTS = []
if SUBJECT_AUTO_TRAIN:
    TRAIN_SCRIPTS.append(_subject_script)

TRAIN_SCRIPTS.extend(
    [
        "precompute_embs.py",
        "precompute_bayesian.py",
        "build_metadata_lookup.py",
        "train_als.py",
        "build_similarity_indices.py",
    ]
)

_VERSIONS_TO_KEEP = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command, printing it first, raising on non-zero exit."""
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def _db_name_from_env() -> str:
    """Resolve the database name from MYSQL_DB or DATABASE_URL."""
    db = os.getenv("MYSQL_DB", "").strip()
    if db:
        return db
    try:
        return (urlsplit(os.getenv("DATABASE_URL", "")).path or "").lstrip("/")
    except Exception:
        return ""


def _read_tail(log_dir: Path) -> str | None:
    """
    Return the tails of all script log files in a version log directory.

    Each file's tail is prefixed with its filename as a header so that the
    crash notification identifies which script produced which output.
    Returns None if the directory does not exist or contains no log files.
    """
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return None
    parts = []
    for log in logs:
        tail = _notify_read_tail(str(log))
        if tail:
            parts.append(f"--- {log.name} ---\n{tail}")
    return "\n\n".join(parts) if parts else None


def _run_local_script(script: str, local_log_dir: Path) -> None:
    """
    Execute a single training script locally and capture its output.

    Output is streamed to stdout in real time and written to a log file
    named after the script stem inside local_log_dir.

    Args:
        script: Filename of the training script (e.g. 'train_als.py').
        local_log_dir: Version-specific local log directory.

    Raises:
        subprocess.CalledProcessError: If the script exits with a non-zero code.
    """
    script_stem = Path(script).stem
    local_log = local_log_dir / f"{script_stem}.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "models/training" / script),
        "--pad-idx",
        str(LOCAL_PAD_IDX),
    ]

    print(f"Running: {' '.join(cmd)}")
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        lines = []
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    with open(local_log, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Execute the full local training pipeline:

    1.  Export training data into local staging/data/.
    2.  Back up the local database.
    3.  Seed staging/attention/ with the current production attention weights
        so training warm-starts from the latest checkpoint.
    4.  Run all training scripts locally, streaming output to stdout and log files.
    5.  Evaluate the quality gate against staging metrics.
    6.  If rejected: notify and exit — production is untouched.
    7.  If approved: promote staging to a versioned directory, signal model
        server containers to reload, flush Redis cache, and retire old versions.
    8.  Update Meilisearch bayes_pop index.

    Any unexpected exception sends a fail notification and re-raises.
    Gate rejections are handled separately so they exit cleanly without
    being treated as crashes.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    version_id = generate_version_id()
    local_log_dir = LOG_DIR / version_id
    local_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run version ID : {version_id}")
    print(f"LOCAL PAD_IDX           : {LOCAL_PAD_IDX}")

    try:
        # --- Data export ----------------------------------------------------

        print("Exporting training data to local staging/data/...")
        run(f"python -m models.training.export_training_data")

        # --- Database backup ------------------------------------------------

        print("Backing up current database...")
        run(f"python {PROJECT_ROOT}/ops/training/backup_db.py")

        # --- Seed staging attention from production -------------------------

        print("Ensuring local staging directory structure exists...")
        PATHS.ensure_staging_dirs()

        print("Seeding staging/attention/ with current production weights...")
        src = PATHS.attention_dir
        dst = PATHS.staging_dir / "attention"
        for pth_file in src.glob("*.pth"):
            shutil.copy2(pth_file, dst / pth_file.name)

        # --- Local training -------------------------------------------------

        print("Running training scripts locally...")
        for script in TRAIN_SCRIPTS:
            print(f"-> {script}")
            _run_local_script(script, local_log_dir)

        # --- Quality gate ---------------------------------------------------

        print("Evaluating deployment quality gate...")
        decision = evaluate()

        print(f"Gate decision: {'APPROVED' if decision.approved else 'REJECTED'}")
        print(f"Reason: {decision.reason}")

        if not decision.approved:
            notify(
                "fail",
                "Training gate rejected — production untouched",
                body=decision.reason,
            )
            print("Gate rejected. Exiting without promoting. Active version unchanged.")
            sys.exit(1)

        # --- Promotion ------------------------------------------------------

        print(f"Promoting staging to version '{version_id}'...")
        manifest = promote_staging(version_id)

        print("Signalling model server containers to reload...")
        signal_workers_reload()

        print("Flushing ALS-dependent Redis cache entries...")
        run(f"python {PROJECT_ROOT}/ops/training/flush_cache.py")

        print(f"Retiring old versions (keeping {_VERSIONS_TO_KEEP})...")
        retire_old_versions(keep=_VERSIONS_TO_KEEP)

        # --- Meilisearch update ---------------------------------------------

        print("Updating bayes_pop in Meilisearch...")
        run(f"python {PROJECT_ROOT}/ops/meilisearch/update_bayes_pop.py")

    except Exception as exc:
        notify(
            "fail",
            "Training pipeline crashed",
            body=f"{type(exc).__name__}: {exc}",
            log_tail=_read_tail(local_log_dir),
        )
        raise

    print(f"Done. Version '{manifest.version_id}' is now active.")
    print(f"Logs saved to: {local_log_dir}")


if __name__ == "__main__":
    main()
