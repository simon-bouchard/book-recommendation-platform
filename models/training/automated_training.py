import subprocess
import json
import os, sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

AZURE_AUTH = os.getenv("AZURE_AUTH_LOCATION")
AZURE_VM = os.getenv("AZURE_VM_NAME")
AZURE_RG = os.getenv("AZURE_RESOURCE_GROUP")
REMOTE_HOST = os.getenv("REMOTE_HOST")
REMOTE_REPO = os.getenv("REMOTE_REPO_PATH")

with open(AZURE_AUTH) as f:
    sp = json.load(f)

AZURE_CLIENT_ID = sp["clientId"]
AZURE_CLIENT_SECRET = sp["clientSecret"]
AZURE_TENANT_ID = sp["tenantId"]

INFERENCE_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TRAINING_DATA_NEW = PROJECT_ROOT / "models/training/data/new_data"
TRAINING_DATA_MAIN = PROJECT_ROOT / "models/training/data"
REMOTE_DATA = f"{REMOTE_HOST}:{REMOTE_REPO}/models/training/data"
REMOTE_MODELS = f"{REMOTE_HOST}:{REMOTE_REPO}/models/data"
LOCAL_MODELS = PROJECT_ROOT / "models/data"
LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR", PROJECT_ROOT / "models/training/logs"))

TRAIN_SCRIPTS = [
    "train_subject_embs.py",
    "precompute_embs.py",
    "precompute_bayesian.py",
    "train_cold_gbt.py",
    "train_als.py",
    "train_warm_gbt.py",
]

def run(cmd, **kwargs):
    print(f"▶ Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_file_local = LOG_DIR / f"{timestamp}_train.log"
    log_file_remote = f"{REMOTE_REPO}/models/training/logs/{timestamp}_train.log"

    # Perform login
    print("📡 Authenticating with Azure...")
    run(f'az login --service-principal -u {AZURE_CLIENT_ID} -p {AZURE_CLIENT_SECRET} --tenant {AZURE_TENANT_ID}')

    print("🚀 Starting training VM...")
    run(f'az vm start --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    print("📦 Exporting training data locally...")
    run("python models/training/export_training_data.py")

    print("⌛ Waiting for SSH to be available...")
    while subprocess.run(f"ssh -o BatchMode=no {REMOTE_HOST} 'echo ready'", shell=True).returncode != 0:
        print("  - Still waiting for SSH...")
        time.sleep(5)

    print("📁 Ensuring remote data directory exists...")
    run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/data'")

    print("📤 Copying new training data to training server...")
    run(f"scp -r {TRAINING_DATA_NEW}/* {REMOTE_DATA}/")

    print("⚙️ Running training scripts remotely...")
    for script in TRAIN_SCRIPTS:
        print(f"➡ {script}")
        cmd = (
            f"ssh {REMOTE_HOST} "
            f"'cd {REMOTE_REPO} && "
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate bookrec-api && "
            f"python models/training/{script}'"
        )
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            lines = []
            for line in proc.stdout:
                print(line, end="")
                lines.append(line)
            # Save logs remotely too
            run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/logs'")
            joined = "".join(lines).replace("'", "'\\''")
            run(f"ssh {REMOTE_HOST} \"echo '{joined}' >> {log_file_remote}\"")
        with open(log_file_local, "a") as f:
            f.writelines(lines)

    print("📥 Copying trained models back...")
    run(f"scp -r {REMOTE_MODELS}/* {LOCAL_MODELS}/")

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