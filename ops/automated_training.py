# ops/automated_training.py
"""
Automated training pipeline that passes local PAD_IDX to remote training scripts.
"""

import subprocess
import json
import os
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import shlex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# Azure config
AZURE_AUTH = os.getenv("AZURE_AUTH_LOCATION")
AZURE_VM = os.getenv("AZURE_VM_NAME")
AZURE_RG = os.getenv("AZURE_RESOURCE_GROUP")
REMOTE_HOST = os.getenv("REMOTE_HOST")
REMOTE_REPO = os.getenv("REMOTE_REPO_PATH")
REMOTE_BACKUP_DIR = os.getenv("REMOTE_BACKUP_DIR")

# Get LOCAL PAD_IDX (from database environment)
LOCAL_PAD_IDX = int(os.getenv("PAD_IDX", "0"))

with open(AZURE_AUTH) as f:
    sp = json.load(f)

AZURE_CLIENT_ID = sp["clientId"]
AZURE_CLIENT_SECRET = sp["clientSecret"]
AZURE_TENANT_ID = sp["tenantId"]

INFERENCE_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAINING_DATA_NEW = PROJECT_ROOT / "models/training/data/new_data"
TRAINING_DATA_MAIN = PROJECT_ROOT / "models/training/data"
REMOTE_DATA = f"{REMOTE_HOST}:{REMOTE_REPO}/models/training/data"
REMOTE_MODELS = f"{REMOTE_HOST}:{REMOTE_REPO}/models/artifacts"
LOCAL_MODELS = PROJECT_ROOT / "models/artifacts"
LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR", PROJECT_ROOT / "models/training/logs"))

ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar").lower()
train_subject_script = os.getenv("SUBJECT_TRAIN_FILE", "train_subject_embs_scalar.py")
SUBJECT_AUTO_TRAIN = os.getenv("SUBJECT_AUTO_TRAIN", "false")

train_subject_script_path = Path(PROJECT_ROOT / "models/training" / train_subject_script)
if not train_subject_script_path.exists():
    train_subject_script = "train_subject_embs_scalar.py"

TRAIN_SCRIPTS = []
if SUBJECT_AUTO_TRAIN.lower() == "true":
    TRAIN_SCRIPTS.append(train_subject_script)

TRAIN_SCRIPTS.extend(
    [
        "precompute_embs.py",
        "precompute_bayesian.py",
        "train_als.py",
    ]
)


def run(cmd, **kwargs):
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_file_local = LOG_DIR / f"{timestamp}_train.log"
    log_file_remote = f"{REMOTE_REPO}/models/training/logs/{timestamp}_train.log"

    print(f"LOCAL PAD_IDX = {LOCAL_PAD_IDX}")
    print("This will be passed to remote training scripts")

    # Azure login
    print("Authenticating with Azure...")
    run(
        f"az login --service-principal -u {AZURE_CLIENT_ID} -p {AZURE_CLIENT_SECRET} --tenant {AZURE_TENANT_ID}"
    )

    print("Starting training VM...")
    run(f'az vm start --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    print("Exporting training data locally...")
    run("python models/training/export_training_data.py")

    print("Waiting for SSH to be available...")
    while (
        subprocess.run(f"ssh -o BatchMode=no {REMOTE_HOST} 'echo ready'", shell=True).returncode
        != 0
    ):
        print("  - Still waiting for SSH...")
        time.sleep(5)

    print("Backing up current database...")
    run(f"python {PROJECT_ROOT}/ops/backup_db.py")

    # Upload backup
    def _db_from_env():
        db = os.getenv("MYSQL_DB", "").strip()
        if db:
            return db
        try:
            from urllib.parse import urlsplit

            return (urlsplit(os.getenv("DATABASE_URL", "")).path or "").lstrip("/")
        except Exception:
            return ""

    db_name = _db_from_env()
    backup_dir = PROJECT_ROOT / "data/backups/db"
    candidates = sorted(backup_dir.glob(f"*_{db_name}.sql.gz"))
    if candidates:
        latest_dump = candidates[-1]
        run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_BACKUP_DIR}'")
        run(f"scp {shlex.quote(str(latest_dump))} {REMOTE_HOST}:{REMOTE_BACKUP_DIR.rstrip('/')}/")

    print("Ensuring remote data directory exists...")
    run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/data'")

    print("Copying new training data to training server...")
    run(f"scp -r {TRAINING_DATA_NEW}/* {REMOTE_DATA}/")

    print("Running training scripts remotely...")
    for script in TRAIN_SCRIPTS:
        print(f"-> {script}")

        # Build command with --pad-idx argument
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

            # Save logs remotely
            run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/logs'")
            joined = "".join(lines).replace("'", "'\\''")
            run(f"ssh {REMOTE_HOST} \"echo '{joined}' >> {log_file_remote}\"")

        with open(log_file_local, "a") as f:
            f.writelines(lines)

    print("Copying trained models back...")
    run(f"scp -r {REMOTE_MODELS}/* {LOCAL_MODELS}/")

    print("Stopping training VM...")
    run(f'az vm deallocate --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    print("Replacing old training data with new data...")
    for file in TRAINING_DATA_MAIN.glob("*"):
        if file.is_file():
            file.unlink()
    for file in TRAINING_DATA_NEW.glob("*"):
        shutil.move(str(file), TRAINING_DATA_MAIN)

    print("Reloading models in memory for all workers...")
    signal_file = Path(PROJECT_ROOT) / "models" / "data" / ".reload_signal"
    with open(signal_file, "w") as f:
        f.write(str(time.time()))

    print("✅ Reload signal written:", signal_file)

    print("Updating bayes_pop in Meilisearch...")
    run(f"python {PROJECT_ROOT}/ops/meilisearch/update_bayes_pop.py")

    print(f"Done. Logs saved to: {log_file_local}")


if __name__ == "__main__":
    main()
