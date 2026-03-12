# ops/training/automated_training.py
"""
Automated training pipeline that orchestrates remote training, artifact staging,
quality gate evaluation, versioned promotion, and worker reload signaling.
"""

import os
import shlex
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
# Remote host configuration
# ---------------------------------------------------------------------------

REMOTE_HOST = os.getenv("REMOTE_HOST")
REMOTE_REPO = os.getenv("REMOTE_REPO_PATH")
REMOTE_BACKUP_DIR = os.getenv("REMOTE_BACKUP_DIR")

# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------

LOCAL_PAD_IDX = int(os.getenv("PAD_IDX", "0"))
LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR", str(PROJECT_ROOT / "models/training/logs")))

# ---------------------------------------------------------------------------
# SSH configuration
# ---------------------------------------------------------------------------

_SSH_READY_TIMEOUT = int(os.getenv("SSH_READY_TIMEOUT", "300"))

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
        "train_als.py",
    ]
)

_VERSIONS_TO_KEEP = 5
_REMOTE_BACKUPS_TO_KEEP = 3
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


def _deallocate_vm() -> None:
    """Deallocate the Azure training VM, logging but not raising on failure."""
    azure_vm = os.getenv("AZURE_VM_NAME")
    azure_rg = os.getenv("AZURE_RESOURCE_GROUP")
    try:
        run(f'az vm deallocate --name {azure_vm} --resource-group "{azure_rg}"')
    except subprocess.CalledProcessError as exc:
        print(f"Warning: VM deallocation failed: {exc}. Continuing.")


def _wait_for_ssh(timeout: int = _SSH_READY_TIMEOUT) -> None:
    """
    Block until the remote host accepts SSH connections, then return.

    Polls every 5 seconds until the host responds or the timeout is exceeded.

    Args:
        timeout: Maximum number of seconds to wait before raising.

    Raises:
        TimeoutError: If the host does not become reachable within timeout seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            f"ssh -o BatchMode=yes -o ConnectTimeout=5 {REMOTE_HOST} 'echo ready'",
            shell=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return
        print("  Still waiting for SSH...")
        time.sleep(5)
    raise TimeoutError(f"Remote host '{REMOTE_HOST}' did not become reachable within {timeout}s.")


def _run_remote_script(
    script: str,
    local_log_dir: Path,
    remote_log_dir: str,
) -> None:
    """
    Execute a single training script on the remote host and capture its output.

    The remote script's output is piped through tee so that it is written to
    the remote log file as it runs. The same output is streamed locally in real
    time and written to the corresponding local log file.

    Log filenames are derived from the script stem, so each script produces its
    own discrete log file in both the local and remote version log directories.

    Args:
        script: Filename of the training script (e.g. 'train_als.py').
        local_log_dir: Version-specific local log directory.
        remote_log_dir: Version-specific remote log directory (absolute path string).

    Raises:
        subprocess.CalledProcessError: If the remote script exits with a non-zero code.
    """
    script_stem = Path(script).stem
    local_log = local_log_dir / f"{script_stem}.log"
    remote_log = f"{remote_log_dir}/{script_stem}.log"

    cmd = (
        f"ssh {REMOTE_HOST} "
        f"'cd {REMOTE_REPO} && "
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate bookrec-api && "
        f"python models/training/{script} --pad-idx {LOCAL_PAD_IDX} 2>&1 "
        f"| tee {remote_log}'"
    )

    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
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
    Execute the full training pipeline:

    1.  Load and validate Azure credentials.
    2.  Authenticate with Azure and start the training VM.
    3.  Export training data into local staging/data/.
    4.  Wait for the VM to accept SSH connections.
    5.  Back up the local database and upload the dump to the remote.
    6.  Copy staging/data/ to the remote staging/data/.
    7.  Prepare the remote artifact staging directory structure.
    8.  Run all training scripts remotely, streaming output locally.
    9.  Pull remote version logs back to local log directory.
    10. SCP produced artifacts into local staging/.
    11. Deallocate the VM (always, regardless of gate outcome).
    12. Evaluate the quality gate against staging metrics.
    13. If rejected: notify and exit — production is untouched.
    14. If approved: promote staging to a versioned directory, write the
        reload signal, and retire old versions.
    15. Update Meilisearch bayes_pop index.

    Any unexpected exception sends a fail notification and re-raises.
    Gate rejections are handled separately so they exit cleanly without
    being treated as crashes.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    version_id = generate_version_id()
    local_log_dir = LOG_DIR / version_id
    local_log_dir.mkdir(parents=True, exist_ok=True)

    remote_log_dir = f"{REMOTE_REPO}/models/training/logs/{version_id}"

    print(f"Training run version ID : {version_id}")
    print(f"LOCAL PAD_IDX           : {LOCAL_PAD_IDX}")

    notify("start", "Training pipeline started", body=f"Version: {version_id}")

    try:
        # --- Azure setup ----------------------------------------------------

        azure_auth_path = os.getenv("AZURE_AUTH_LOCATION")
        if not azure_auth_path:
            raise RuntimeError(
                "AZURE_AUTH_LOCATION is not set. "
                "Ensure the environment variable points to the service principal JSON file."
            )

        import json

        with open(azure_auth_path) as f:
            sp = json.load(f)

        azure_client_id = sp["clientId"]
        azure_client_secret = sp["clientSecret"]
        azure_tenant_id = sp["tenantId"]
        azure_vm = os.getenv("AZURE_VM_NAME")
        azure_rg = os.getenv("AZURE_RESOURCE_GROUP")

        print("Authenticating with Azure...")
        run(
            f"az login --service-principal "
            f"-u {azure_client_id} -p {azure_client_secret} --tenant {azure_tenant_id}"
        )

        print("Starting training VM...")
        run(f'az vm start --name {azure_vm} --resource-group "{azure_rg}"')

        # --- Data preparation -----------------------------------------------

        print("Exporting training data to local staging/data/...")
        run(f"python -m models.training.export_training_data")

        print("Waiting for SSH to be available...")
        _wait_for_ssh()

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

        print("Copying staging/data/ to remote staging/data/...")
        remote_staging = f"{REMOTE_REPO}/models/artifacts/staging"
        run(f"ssh {REMOTE_HOST} 'mkdir -p {remote_staging}/data'")
        run(f"scp -r {PATHS.staging_data_dir}/ {REMOTE_HOST}:{remote_staging}/data/")

        print("Ensuring remote artifact staging structure is ready...")
        run(
            f"ssh {REMOTE_HOST} "
            f"'mkdir -p "
            f"{remote_staging}/embeddings "
            f"{remote_staging}/attention "
            f"{remote_staging}/scoring "
            f"{remote_log_dir}'"
        )

        # --- Remote training ------------------------------------------------

        print("Running training scripts remotely...")
        for script in TRAIN_SCRIPTS:
            print(f"-> {script}")
            _run_remote_script(script, local_log_dir, remote_log_dir)

        # --- Log retrieval --------------------------------------------------

        print("Pulling remote version logs to local log directory...")
        run(f"scp -r {REMOTE_HOST}:{remote_log_dir}/ {local_log_dir}/")

        # --- Artifact retrieval ---------------------------------------------

        print("Ensuring local staging directory exists...")
        PATHS.ensure_staging_dirs()

        print("Copying trained artifacts into local staging...")
        for subdir in ("embeddings", "attention", "scoring"):
            run(f"scp -r {REMOTE_HOST}:{remote_staging}/{subdir} {PATHS.staging_dir}/")
        run(f"scp {REMOTE_HOST}:{remote_staging}/training_metrics.json {PATHS.staging_dir}/")

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

        print("Signalling model server containers to reload...")
        signal_workers_reload()

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

    # --- Success notification -----------------------------------------------

    if _NOTIFY_ON_SUCCESS:
        recall_str = f"Recall@30={decision.staging_recall:.4f}" if decision.staging_recall else ""
        notify(
            "ok",
            "Training pipeline succeeded",
            body=f"Version: {manifest.version_id}  {recall_str}",
        )

    print(f"Done. Version '{manifest.version_id}' is now active.")
    print(f"Logs saved to: {local_log_dir}")


if __name__ == "__main__":
    main()
