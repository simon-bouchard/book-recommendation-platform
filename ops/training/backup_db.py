#!/usr/bin/env python3
import datetime
import os
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = REPO_ROOT / "data" / "backups" / "db"
TS = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_mysql_url(url: str):
    try:
        u = urlsplit(url)
        return (
            (u.username or ""),
            (u.password or ""),
            (u.hostname or "127.0.0.1"),
            (u.path or "").lstrip("/"),
        )
    except Exception:
        return "", "", "127.0.0.1", ""


def run(cmd: str, env=None):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)


def main():
    db_url = os.getenv("DATABASE_URL", "")
    user = os.getenv("MYSQL_USER", "")
    pwd = os.getenv("MYSQL_PASSWORD", "")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    db = os.getenv("MYSQL_DB", "")

    if not (user and pwd and db):
        u, p, h, d = parse_mysql_url(db_url)
        user = user or u
        pwd = pwd or p
        host = host or h
        db = db or d

    if not (user and pwd and host and db):
        raise SystemExit("MySQL dump: missing creds (set MYSQL_* or DATABASE_URL).")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = LOCAL_DIR / f"{TS}_{db}.sql.gz"

    env = os.environ.copy()
    env["MYSQL_PWD"] = pwd

    # Avoid PROCESS privilege requirement
    dump_cmd = (
        f"mysqldump --single-transaction --quick --triggers --routines --no-tablespaces "
        f"-h {shlex.quote(host)} -u {shlex.quote(user)} {shlex.quote(db)} | gzip -c > {shlex.quote(str(dump_path))}"
    )
    run(dump_cmd, env=env)

    print(str(dump_path))

    dumps = sorted(LOCAL_DIR.glob(f"*_{db}.sql.gz"))
    for old in dumps[:-7]:
        try:
            old.unlink()
        except:
            pass


if __name__ == "__main__":
    main()
