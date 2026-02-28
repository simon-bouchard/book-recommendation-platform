# ops/training/automated_training.py
"""
Automated training pipeline that orchestrates remote training, artifact staging,
quality gate evaluation, versioned promotion, and worker reload signaling.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from models.core.artifact_registry import (
    _write_reload_signal,
    generate_version_id,
    promote_staging,
    retire_old_versions,
)
from models.core.paths import PATHS
from ops.training.evaluate_gate import evaluate
from ops.training.notify import notify, read_tail as _notify_read_tail

# ---------------------------------------------------------------------------
# Azure configuration
# ---------------------------------------------------------------------------

AZURE_AUTH = os.getenv("AZURE_AUTH_LOCATION")
AZURE_VM = os.getenv("AZURE_VM_NAME")
AZURE_RG = os.getenv("AZURE_RESOURCE_GROUP")

with open(AZURE_AUTH) as f:
    _sp = json.load(f)

AZURE_CLIENT_ID = _sp["clientId"]
AZURE_CLIENT_SECRET = _sp["clientSecret"]
AZURE_TENANT_ID = _sp["tenantId"]

# ---------------------------------------------------------------------------
# Remote host configuration
# ---------------------------------------------------------------------------

REMOTE_HOST = os.getenv("REMOTE_HOST")
REMOTE_REPO = os.getenv("REMOTE_REPO_PATH")
REMOTE_BACKUP_DIR = os.getenv("REMOTE_BACKUP_DIR")
REMOTE_ARTIFACTS = f"{REMOTE_HOST}:{REMOTE_REPO}/models/artifacts"

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------

LOCAL_PAD_IDX = int(os.getenv("PAD_IDX", "0"))
TRAINING_DATA_NEW = PROJECT_ROOT / "models/training/data/new_data"
TRAINING_DATA_MAIN = PROJECT_ROOT / "models/training/data"
REMOTE_DATA = f"{REMOTE_HOST}:{REMOTE_REPO}/models/training/data"
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

TRAIN_SCRIPTS.extend([
    "precompute_embs.py",
    "precompute_bayesian.py",
    "train_als.py",
])

_VERSIONS_TO_KEEP = 5
_REMOTE_BACKUPS_TO_KEEP = 3

# Set to false once you've confirmed the pipeline is working reliably.
_NOTIFY_ON_SUCCESS = os.getenv("NOTIFY_ON_SUCCESS", "true").lower() != "false"


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


def _read_tail(path: Path) -> str | None:
    """Return the last 60 lines of a log file, or None if unreadable."""
    return _notify_read_tail(str(path) if path.exists() else None)


def _deallocate_vm() -> None:
    """Deallocate the Azure training VM, logging but not raising on failure."""
    try:
        run(f'az vm deallocate --name {AZURE_VM} --resource-group "{AZURE_RG}"')
    except subprocess.CalledProcessError as exc:
        print(f"Warning: VM deallocation failed: {exc}. Continuing.")


def _run_remote_script(script: str, log_file_local: Path, log_file_remote: str) -> None:
    """
    Execute a single training script on the remote host and capture its output.

    Streams stdout/stderr locally in real time, then appends to both the local
    log file and the remote log file.
    """
    cmd = (
        f"ssh {REMOTE_HOST} "
        f"'cd {REMOTE_REPO} && "
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate bookrec-api && "
        f"python models/training/{script} --pad-idx {LOCAL_PAD_IDX}'"
    )

    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        lines = []
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)

        run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/logs'")
        joined = "".join(lines).replace("'", "'\\''")
        run(f"ssh {REMOTE_HOST} \"echo '{joined}' >> {log_file_remote}\"")

    with open(log_file_local, "a") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Execute the full training pipeline:

    1.  Authenticate with Azure and start the training VM.
    2.  Export and upload training data.
    3.  Run all training scripts remotely.
    4.  SCP produced artifacts into local staging/.
    5.  Deallocate the VM (always, regardless of gate outcome).
    6.  Evaluate the quality gate against staging metrics.
    7.  If rejected: notify and exit — production is untouched.
    8.  If approved: promote staging to a versioned directory, write the
        reload signal, retire old versions, and notify success.
    9.  Replace local training data with the new export.
    10. Update Meilisearch bayes_pop index.

    Any unexpected exception sends a fail notification and re-raises.
    Gate rejections are handled separately before the try/except so they
    exit cleanly without being treated as crashes.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_file_local = LOG_DIR / f"{timestamp}_train.log"
    log_file_remote = f"{REMOTE_REPO}/models/training/logs/{timestamp}_train.log"

    version_id = generate_version_id()

    print(f"Training run version ID : {version_id}")
    print(f"LOCAL PAD_IDX           : {LOCAL_PAD_IDX}")

    notify("start", "Training pipeline started", body=f"Version: {version_id}")

    try:
        # --- Azure setup ----------------------------------------------------

        print("Authenticating with Azure...")
        run(
            f"az login --service-principal "
            f"-u {AZURE_CLIENT_ID} -p {AZURE_CLIENT_SECRET} --tenant {AZURE_TENANT_ID}"
        )

        print("Starting training VM...")
        run(f'az vm start --name {AZURE_VM} --resource-group "{AZURE_RG}"')

        # --- Data preparation -----------------------------------------------

        print("Exporting training data locally...")
        run(f"python {PROJECT_ROOT}/models/training/export_training_data.py")

        print("Waiting for SSH to be available...")
        while subprocess.run(
            f"ssh -o BatchMode=yes {REMOTE_HOST} 'echo ready'", shell=True
        ).returncode != 0:
            print("  Still waiting for SSH...")
            time.sleep(5)

        print("Backing up current database...")
        run(f"python {PROJECT_ROOT}/ops/training/backup_db.py")

        db_name = _db_name_from_env()
        backup_dir = PROJECT_ROOT / "data/backups/db"
        candidates = sorted(backup_dir.glob(f"*_{db_name}.sql.gz"))
        if candidates:
            latest_dump = candidates[-1]
            run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_BACKUP_DIR}'")
            run(
                f"scp {shlex.quote(str(latest_dump))} "
                f"{REMOTE_HOST}:{REMOTE_BACKUP_DIR.rstrip('/')}/"
            )
            run(
                f"ssh {REMOTE_HOST} "
                f"'ls -t {REMOTE_BACKUP_DIR}/*.sql.gz 2>/dev/null "
                f"| tail -n +{_REMOTE_BACKUPS_TO_KEEP + 1} | xargs -r rm -f'"
            )

        print("Copying new training data to training server...")
        run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/data'")
        run(f"scp -r {TRAINING_DATA_NEW}/* {REMOTE_DATA}/")

        # --- Remote training ------------------------------------------------

        print("Running training scripts remotely...")
        for script in TRAIN_SCRIPTS:
            print(f"-> {script}")
            _run_remote_script(script, log_file_local, log_file_remote)

        # --- Artifact retrieval ---------------------------------------------

        print("Ensuring local staging directory exists...")
        PATHS.ensure_staging_dirs()

        print("Copying trained artifacts into staging...")
        run(f"scp -r {REMOTE_ARTIFACTS}/* {PATHS.staging_dir}/")

        # --- VM deallocation (always runs) ----------------------------------

        print("Deallocating training VM...")
        _deallocate_vm()

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

        print("Writing reload signal for Gunicorn workers...")
        _write_reload_signal()

        print(f"Retiring old versions (keeping {_VERSIONS_TO_KEEP})...")
        retire_old_versions(keep=_VERSIONS_TO_KEEP)

        # --- Training data rotation -----------------------------------------

        print("Replacing local training data with new export...")
        for file in TRAINING_DATA_MAIN.glob("*"):
            if file.is_file():
                file.unlink()
        for file in TRAINING_DATA_NEW.glob("*"):
            shutil.move(str(file), TRAINING_DATA_MAIN)

        # --- Meilisearch update ---------------------------------------------

        print("Updating bayes_pop in Meilisearch...")
        run(f"python {PROJECT_ROOT}/ops/meilisearch/update_bayes_pop.py")

    except Exception as exc:
        notify(
            "fail",
            "Training pipeline crashed",
            body=f"{type(exc).__name__}: {exc}",
            log_tail=_read_tail(log_file_local),
        )
        raise

    # --- Success notification -----------------------------------------------

    if _NOTIFY_ON_SUCCESS:
        recall_str = (
            f"Recall@30={decision.staging_recall:.4f}" if decision.staging_recall else ""
        )
        notify(
            "ok",
            "Training pipeline succeeded",
            body=f"Version: {manifest.version_id}  {recall_str}",
        )

    print(f"Done. Version '{manifest.version_id}' is now active.")
    print(f"Logs saved to: {log_file_local}")


if __name__ == "__main__":
    main()
