#!/usr/bin/env python3
import os, subprocess, datetime, shlex
from pathlib import Path
from urllib.parse import urlsplit

ENV_FILE = "/etc/bookrec.env"
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../book_recommendation_api
LOCAL_DIR = REPO_ROOT / "data" / "backups" / "db"

TS = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def load_env_file(p):
    if not os.path.exists(p): return
    with open(p) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s: continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k, v)

def parse_mysql_url(url: str):
    try:
        u = urlsplit(url)
        return (u.username or ""), (u.password or ""), (u.hostname or "127.0.0.1"), (u.path or "").lstrip("/")
    except Exception:
        return "", "", "127.0.0.1", ""

def run(cmd: str, env=None):
    subprocess.run(cmd, shell=True, check=True, env=env)

def main():
    load_env_file(ENV_FILE)

    db_url = os.getenv("DATABASE_URL", "")
    user = os.getenv("MYSQL_BACKUP_USER", "") or os.getenv("MYSQL_USER", "")
    pwd  = os.getenv("MYSQL_BACKUP_PASSWORD", "") or os.getenv("MYSQL_PASSWORD", "")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    db   = os.getenv("MYSQL_DB", "")

    if not (user and pwd and db):
        u, p, h, d = parse_mysql_url(db_url)
        user = user or u
        pwd  = pwd or p
        host = host or h
        db   = db or d

    if not (user and pwd and host and db):
        raise SystemExit("MySQL dump: missing creds (set MYSQL_* or DATABASE_URL).")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = LOCAL_DIR / f"{TS}_{db}.sql.gz"

    env = os.environ.copy()
    env["MYSQL_PWD"] = pwd

    dump_cmd = (
        f"mysqldump --single-transaction --quick --triggers "
        f"-h {shlex.quote(host)} -u {shlex.quote(user)} {shlex.quote(db)} | gzip -c > {shlex.quote(str(dump_path))}"
    )
    run(dump_cmd, env=env)

    t_user = os.getenv("TRAINING_SERVER_USER", "REPLACE")
    t_ip   = os.getenv("TRAINING_SERVER_IP", "REPLACE")
    t_dir  = os.getenv("REMOTE_BACKUP_DIR", "~/bookrec_backups")

    if "REPLACE" not in (t_user, t_ip):
        run(f"ssh {t_user}@{t_ip} 'mkdir -p {shlex.quote(t_dir)}'")
        run(f"scp {shlex.quote(str(dump_path))} {t_user}@{t_ip}:{t_dir}/")

    # keep last 7 local dumps
    dumps = sorted(LOCAL_DIR.glob(f"*_{db}.sql.gz"))
    for old in dumps[:-7]:
        try: old.unlink()
        except: pass

if __name__ == "__main__":
    main()

