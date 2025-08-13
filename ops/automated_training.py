import subprocess, shlex
import json
import os, sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print("PROJECT_ROOT:", PROJECT_ROOT)
PY = sys.executable

sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

SSH_CFG  = os.getenv("SSH_CONFIG", "/home/simon/.ssh/config")
SSH_OPTS = f"-F {shlex.quote(SSH_CFG)} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
REMOTE_PY = os.getenv("REMOTE_PY")
REMOTE_REPO = os.getenv("REMOTE_REPO")
REMOTE_LOG_DIR = f"{REMOTE_REPO}/models/training/logs"
REMOTE_BACKUP_DIR = os.getenv("REMOTE_BACKUP_DIR")

AZURE_AUTH = os.getenv("AZURE_AUTH_LOCATION")
AZURE_VM = os.getenv("AZURE_VM_NAME")
AZURE_RG = os.getenv("AZURE_RESOURCE_GROUP")
REMOTE_HOST = os.getenv("REMOTE_HOST")

with open(AZURE_AUTH) as f:
    sp = json.load(f)

AZURE_CLIENT_ID = sp["clientId"]
AZURE_CLIENT_SECRET = sp["clientSecret"]
AZURE_TENANT_ID = sp["tenantId"]

INFERENCE_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DATA_NEW = PROJECT_ROOT / "models/training/data/new_data"
TRAINING_DATA_MAIN = PROJECT_ROOT / "models/training/data"
REMOTE_DATA = f"{REMOTE_HOST}:{REMOTE_REPO}/models/training/data"
REMOTE_MODELS = f"{REMOTE_HOST}:{REMOTE_REPO}/models/data"
LOCAL_MODELS = PROJECT_ROOT / "models/data"
LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR", PROJECT_ROOT / "models/training/logs"))

# Pick the correct train_subject_embs script based on ATTN_STRATEGY
ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar").lower()
train_subject_script = f"train_subject_embs_{ATTN_STRATEGY}.py"

# Fallback if script doesn’t exist (e.g., for scalar)
train_subject_script_path = Path(PROJECT_ROOT / "models/training" / train_subject_script)
if not train_subject_script_path.exists():
    train_subject_script = "train_subject_embs_scalar.py"

BASE_ENV = os.environ.copy()
# ensure top-level packages (app/, models/, routes/, etc.) import reliably
BASE_ENV["PYTHONPATH"] = str(PROJECT_ROOT)
BASE_ENV.setdefault("HOME", "/home/simon")

TRAIN_SCRIPTS = [
    train_subject_script,
    "precompute_embs.py",
    "precompute_bayesian.py",
    "train_cold_gbt.py",
    "train_als.py",
    "train_warm_gbt.py",
]

def run(cmd, **kwargs):
    # string -> shell, list -> argv (unchanged behavior)
    kwargs.setdefault("env", BASE_ENV)
    shown = cmd if isinstance(cmd, str) else " ".join(map(shlex.quote, cmd))
    print(f"▶ Running: {shown}")
    p = subprocess.run(cmd, shell=isinstance(cmd, str), text=True,
                       capture_output=True, **kwargs)
    if p.stdout: print(p.stdout, end="")
    if p.returncode != 0:
        if p.stderr: print(p.stderr, end="")
        raise subprocess.CalledProcessError(p.returncode, shown)
    return 0

def run_py(rel_script, args=None):
    # *** prepend repo root → absolute path ***
    script_abs = (PROJECT_ROOT / rel_script).as_posix()
    if not os.path.exists(script_abs):
        print(f"!! Not found: {script_abs}")
        raise SystemExit(2)
    argv = [PY, script_abs]
    if args:
        argv += (shlex.split(args) if isinstance(args, str) else list(args))
    return run(argv) 

def run_remote_in_repo(cmd):
    # login shell, CD into repo, then run the command
    run(f"ssh {SSH_OPTS} {REMOTE_HOST} bash -lc {shlex.quote(f'cd {REMOTE_REPO} && {cmd}')}")

def run_remote(cmd: str):
    # Direct remote exec. Do NOT use bash -lc here.
    run(f"ssh {SSH_OPTS} {REMOTE_HOST} {cmd}")

def main():
    print("SSH alias:", REMOTE_HOST)
    print(REMOTE_REPO)
    run(f"ssh -G {SSH_OPTS} {REMOTE_HOST} | egrep '^(user|hostname|identityfile) '")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_file_local = LOG_DIR / f"{timestamp}_train.log"
    log_file_remote = f"{REMOTE_LOG_DIR}/{timestamp}_train.log"

    print("SSH alias:", REMOTE_HOST)
    run(f"ssh -G {SSH_OPTS} {REMOTE_HOST} | egrep '^(user|hostname|identityfile) '")

    # Perform login
    print("📡 Authenticating with Azure...")
    run(f'az login --service-principal -u {AZURE_CLIENT_ID} -p {AZURE_CLIENT_SECRET} --tenant {AZURE_TENANT_ID}')

    print("🚀 Starting training VM...")
    run(f'az vm start --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    print("📦 Exporting training data locally...")
    run_py("models/training/export_training_data.py")

    print("⌛ Waiting for SSH to be available...")
    attempt = 0
    max_attempts = 20
    while subprocess.run(f"ssh {SSH_OPTS} {REMOTE_HOST} 'echo ready'", shell=True).returncode != 0:
        attempt += 1
        if attempt >= max_attempts:
            raise RuntimeError("SSH not reachable after 20 attempts")
        print("  - Still waiting for SSH...")
        time.sleep(5)

    print("🔄 Backing up current database...")
    run_py("ops/backup_db.py")

    # Ship latest DB dump to training VM
    from urllib.parse import urlsplit
    def _db_from_env():
        db = os.getenv("MYSQL_DB", "").strip()
        if db:
            return db
        try:
            return (urlsplit(os.getenv("DATABASE_URL","")).path or "").lstrip("/")
        except Exception:
            return ""
    db_name = _db_from_env()
    backup_dir = PROJECT_ROOT / "data/backups/db"
    candidates = sorted(backup_dir.glob(f"*_{db_name}.sql.gz"))
    if candidates:
        latest_dump = candidates[-1]
        run_remote(f"mkdir -p -- {shlex.quote(REMOTE_BACKUP_DIR)}")
        run(f"scp {SSH_OPTS} {shlex.quote(str(latest_dump))} {REMOTE_HOST}:{REMOTE_BACKUP_DIR.rstrip('/')}/")
     
    print("📁 Ensuring remote log directory exists...")
    run_remote(f"mkdir -p -- {shlex.quote(REMOTE_LOG_DIR)}")
    
    print("📤 Copying new training data to training server...")
    run(f"scp -r {SSH_OPTS} {TRAINING_DATA_NEW}/. {REMOTE_HOST}:{shlex.quote(REMOTE_REPO)}/models/training/data/")

    print("⚙️ Running training scripts remotely...")
    for script in TRAIN_SCRIPTS:
        print(f"➡ {script}")
        # run from the repo so relative paths like models/... resolve correctly
        remote_script = shlex.quote(f"{REMOTE_REPO}/models/training/{script}")
        run_remote_in_repo(
            f"{REMOTE_PY} {remote_script} 2>&1 | tee -a {shlex.quote(log_file_remote)}"
        ) 

    print("📥 Copying trained models back...")
    run(f"scp -r {SSH_OPTS} {REMOTE_HOST}:{shlex.quote(REMOTE_REPO)}/models/data/. {LOCAL_MODELS}/")

    print("🛑 Stopping training VM...")
    run(f'az vm deallocate --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    print("📁 Replacing old training data with new data...")
    for file in TRAINING_DATA_MAIN.glob("*"):
        if file.is_file():
            file.unlink()
    for file in TRAINING_DATA_NEW.glob("*"):
        shutil.move(str(file), TRAINING_DATA_MAIN)

    print("🧠 Reloading models in memory...")
    api_url = os.getenv("RELOAD_API_URL", "http://localhost:8000/admin/reload_models")
    admin_secret = os.getenv("ADMIN_SECRET")

    try:
        resp = requests.post(api_url, params={"secret": admin_secret})
        if resp.status_code == 200:
            print("✅ API reload successful.")
        else:
            print(f"❌ API reload failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"❌ Exception during reload: {e}")

    print("✅ Done. Logs saved to:", log_file_local)

if __name__ == "__main__":
    main()
