#!/usr/bin/env python3
# app/enrichment/ol_viaf_subset.py

import bz2
import csv
import gzip
import json
import os
import re
import sys
from typing import Dict, Optional, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.database import SessionLocal, engine
from app.table_models import Author

DUMP_PATH = "data/ol_dump_authors.txt.gz"
OUT_JSON = "data/ol2viaf_subset.json"
OUT_CSV = "data/ol2viaf_subset.csv"

OL_KEY_RE = re.compile(r"^OL\d+A$")
VIAF_URL_RE = re.compile(r"/viaf/(\d+)", re.IGNORECASE)


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def norm_ol_author_key(val: Optional[str]) -> Optional[str]:
    """
    Accepts:
      'OL12345A'
      '/authors/OL12345A'
      '/authors/OL12345A/...'
    Returns 'OL12345A' or None.
    """
    if not val:
        return None
    s = str(val).strip()
    if not s:
        return None
    if "/authors/" in s:
        s = s.split("/authors/")[-1]
    s = s.split("/")[0].strip()
    return s if OL_KEY_RE.fullmatch(s) else None


def load_db_ol_keys() -> Set[str]:
    """Load OL author keys from your DB (authors.external_id)."""
    keys: Set[str] = set()
    db = SessionLocal()
    try:
        for (ext_id,) in db.query(Author.external_id).all():
            k = norm_ol_author_key(ext_id)
            if k:
                keys.add(k)
    finally:
        db.close()
    return keys


def collect_viaf_ids(data_raw: str, data: dict):
    viafs = []

    # 1. Known structured fields
    remote = data.get("remote_ids") or {}
    v = remote.get("viaf")
    if isinstance(v, list):
        viafs.extend(str(x) for x in v if x)
    elif isinstance(v, (str, int)):
        viafs.append(str(v))

    identifiers = data.get("identifiers") or {}
    v = identifiers.get("viaf")
    if isinstance(v, list):
        viafs.extend(str(x) for x in v if x)

    links = data.get("links")
    if isinstance(links, list):
        for link in links:
            url = (link.get("url") or "").lower()
            if "viaf.org" in url:
                m = re.search(r"/viaf/(\d+)", url)
                if m:
                    viafs.append(m.group(1))

    # 2. Fallback: regex over raw JSON line
    m = re.findall(r"viaf[^\d]*(\d+)", data_raw, flags=re.IGNORECASE)
    viafs.extend(m)

    # Deduplicate & return
    return list(dict.fromkeys(viafs))


def parse_dump_for_subset(dump_path: str, target_keys: Set[str]) -> Dict[str, str]:
    """
    Stream the OL authors dump, returning {OL_KEY: VIAF_ID}
    ONLY for keys in target_keys. Stops early when all found.
    Uses a robust VIAF collector (remote_ids/identifiers/links + regex fallback).
    """
    found: Dict[str, str] = {}
    if not target_keys:
        return found

    with open_text(dump_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            # The dump is TSV; JSON is the last column
            parts = line.split("\t")
            data_raw = parts[-1]
            try:
                data = json.loads(data_raw)
            except Exception:
                continue

            key = data.get("key") or ""
            if not key.startswith("/authors/"):
                continue
            ol_key = key.rsplit("/", 1)[-1]  # 'OL12345A'

            if ol_key not in target_keys or ol_key in found:
                continue

            # Robust collector over known fields + regex fallback
            viaf_ids = collect_viaf_ids(data_raw, data)
            if viaf_ids:
                # choose a stable representative (smallest numeric)
                viaf_ids = [
                    re.search(r"\d+", str(v)).group(0)
                    for v in viaf_ids
                    if re.search(r"\d+", str(v))
                ]
                if not viaf_ids:
                    continue
                viaf_id = min(set(viaf_ids), key=int)  # stable rep: smallest numeric
                found[ol_key] = viaf_id
                if len(found) == len(target_keys):
                    break
    return found


def write_json(mapping: Dict[str, str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def write_csv(mapping: Dict[str, str], out_path: str):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ol_author_key", "viaf_id"])
        for k, v in sorted(mapping.items()):
            w.writerow([k, v])


def main():
    if engine is None:
        print("DATABASE_URL is not configured. Check your .env.", file=sys.stderr)
        sys.exit(2)

    target = load_db_ol_keys()
    total_targets = len(target)

    mapping = parse_dump_for_subset(DUMP_PATH, target)

    have_viaf = len(mapping)
    coverage = (have_viaf / total_targets * 100.0) if total_targets else 0.0

    write_json(mapping, OUT_JSON)
    write_csv(mapping, OUT_CSV)

    print(f"Total authors in DB with OL key: {total_targets}")
    print(f"Authors with VIAF: {have_viaf}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"Wrote JSON to: {OUT_JSON}")
    print(f"Wrote CSV  to: {OUT_CSV}")


if __name__ == "__main__":
    main()
