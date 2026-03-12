# models/core/artifact_registry.py
"""
Artifact lifecycle management for versioned model artifacts.

Owns the active_version pointer file, version promotion, rollback,
manifest writing, checksum verification, and version retirement.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from models.core.paths import PATHS, _VERSIONED_SUBDIRS

_MANIFEST_FILENAME = "manifest.json"
_METRICS_FILENAME = "training_metrics.json"
_EXPORT_SENTINEL = ".export_complete"
_CHECKSUM_BLOCK_SIZE = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class VersionManifest:
    """
    Complete record of a promoted model version.

    All fields are written to manifest.json at promotion time and are read
    back verbatim by list_versions(), get_manifest(), and evaluate_gate.
    """

    version_id: str
    created_at: str
    git_commit: Optional[str]
    pad_idx: int
    attn_strategy: str
    metrics: Dict
    checksums: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to a plain dict suitable for JSON encoding."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "VersionManifest":
        """Deserialize from a parsed JSON dict."""
        return cls(
            version_id=data["version_id"],
            created_at=data["created_at"],
            git_commit=data.get("git_commit"),
            pad_idx=data["pad_idx"],
            attn_strategy=data["attn_strategy"],
            metrics=data.get("metrics", {}),
            checksums=data.get("checksums", {}),
        )


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def generate_version_id() -> str:
    """
    Generate a unique, chronologically sortable version ID.

    Format: YYYYMMDD-HHMM-{6-char git hash}.
    Falls back to a random hex suffix if git is unavailable.

    Returns:
        Version ID string, e.g. '20260226-1430-a3f9b2'.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    suffix = _git_short_hash() or _random_hex(6)
    return f"{timestamp}-{suffix}"


# ---------------------------------------------------------------------------
# Core lifecycle operations
# ---------------------------------------------------------------------------


def register_existing_version(
    source_dir: Path,
    version_id: str,
    metrics: Optional[Dict] = None,
) -> VersionManifest:
    """
    Copy an artifact directory into the versioned store and make it active.

    This is the single primitive that both promote_staging() and the migration
    script use. It copies source_dir into versions/{version_id}/, computes
    SHA-256 checksums for every file, writes the manifest, then atomically
    updates the active_version pointer.

    Args:
        source_dir: Directory containing the three versioned subdirs
                    (embeddings/, attention/, scoring/).
        version_id: Version ID to register under.
        metrics: Optional metrics dict to embed in the manifest. If None,
                 training_metrics.json is read from source_dir if present.

    Returns:
        The written VersionManifest.

    Raises:
        FileExistsError: If a version with this ID already exists.
        FileNotFoundError: If source_dir does not exist.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: '{source_dir}'")

    dest_dir = PATHS.versions_dir / version_id
    if dest_dir.exists():
        raise FileExistsError(
            f"Version '{version_id}' already exists at '{dest_dir}'. "
            "Delete it manually or use a different version ID."
        )

    PATHS.versions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying artifacts from '{source_dir}' to '{dest_dir}'...")
    shutil.copytree(source_dir, dest_dir)

    checksums = _compute_checksums(dest_dir)
    resolved_metrics = metrics if metrics is not None else _load_source_metrics(source_dir)

    manifest = VersionManifest(
        version_id=version_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_short_hash(),
        pad_idx=int(os.getenv("PAD_IDX", "0")),
        attn_strategy=os.getenv("ATTN_STRATEGY", "scalar").lower(),
        metrics=resolved_metrics,
        checksums=checksums,
    )

    _write_manifest(dest_dir, manifest)
    _atomic_write_pointer(version_id)

    print(f"Version '{version_id}' registered and set as active.")
    return manifest


def promote_staging(version_id: str) -> VersionManifest:
    """
    Promote the current staging directory to a named version and activate it.

    Verifies that all expected subdirectories are present in staging before
    copying. Reads training_metrics.json from staging for the manifest.

    Args:
        version_id: Version ID to assign to this promotion.

    Returns:
        The written VersionManifest.

    Raises:
        RuntimeError: If any expected subdirectory is missing from staging.
        FileExistsError: If this version ID has already been registered.
    """
    _assert_staging_complete()
    return register_existing_version(PATHS.staging_dir, version_id)


def rollback(version_id: str) -> None:
    """
    Switch the active version to a previously registered version.

    Updates the active_version pointer file only. The caller is responsible
    for triggering a worker reload after rollback so that the new active
    version takes effect in production.

    Args:
        version_id: The version ID to roll back to.

    Raises:
        FileNotFoundError: If the requested version does not exist.
    """
    version_dir = PATHS.versions_dir / version_id
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Version '{version_id}' not found at '{version_dir}'. "
            f"Available versions: {[v.version_id for v in list_versions()]}"
        )

    _atomic_write_pointer(version_id)

    print(f"Rolled back to version '{version_id}'.")


def list_versions() -> List[VersionManifest]:
    """
    Return all registered versions sorted by creation date, newest first.

    Versions whose manifest.json is missing or malformed are silently
    skipped to avoid breaking the listing when artifacts are partially corrupt.

    Returns:
        List of VersionManifest objects sorted descending by created_at.
    """
    if not PATHS.versions_dir.exists():
        return []

    manifests: List[VersionManifest] = []
    for version_dir in PATHS.versions_dir.iterdir():
        if not version_dir.is_dir():
            continue
        manifest_path = version_dir / _MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            manifest = _read_manifest(version_dir)
            manifests.append(manifest)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    manifests.sort(key=lambda m: m.created_at, reverse=True)
    return manifests


def get_manifest(version_id: str) -> VersionManifest:
    """
    Read and return the manifest for a specific version.

    Args:
        version_id: The version ID to look up.

    Returns:
        Parsed VersionManifest.

    Raises:
        FileNotFoundError: If the version directory or manifest does not exist.
        json.JSONDecodeError: If the manifest is not valid JSON.
    """
    version_dir = PATHS.versions_dir / version_id
    manifest_path = version_dir / _MANIFEST_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for version '{version_id}' at '{manifest_path}'."
        )

    return _read_manifest(version_dir)


def retire_old_versions(keep: int = 5) -> List[str]:
    """
    Delete old versions beyond the most recent N, with protected exceptions.

    Always retains:
        - The currently active version.
        - The version immediately preceding the active version (for fast rollback).
        - The most recent `keep` versions by creation date.

    Args:
        keep: Minimum number of most-recent versions to retain. Defaults to 5.

    Returns:
        List of version IDs that were deleted.

    Raises:
        ValueError: If keep < 2, since at least active + one prior must be kept.
    """
    if keep < 2:
        raise ValueError("keep must be at least 2 to protect active and its predecessor.")

    try:
        active_id = PATHS.active_version_id()
    except (FileNotFoundError, ValueError):
        active_id = None

    all_versions = list_versions()  # sorted newest first

    protected: set = set()
    if active_id:
        protected.add(active_id)
        active_index = next(
            (i for i, m in enumerate(all_versions) if m.version_id == active_id), None
        )
        if active_index is not None and active_index + 1 < len(all_versions):
            protected.add(all_versions[active_index + 1].version_id)

    to_keep = {m.version_id for m in all_versions[:keep]}
    to_keep.update(protected)

    retired: List[str] = []
    for manifest in all_versions:
        if manifest.version_id in to_keep:
            continue
        version_dir = PATHS.versions_dir / manifest.version_id
        shutil.rmtree(version_dir)
        retired.append(manifest.version_id)
        print(f"Retired version '{manifest.version_id}'.")

    return retired


def verify_checksums(version_id: str) -> Dict[str, bool]:
    """
    Verify the SHA-256 checksum of every artifact file in a version.

    Args:
        version_id: The version to verify.

    Returns:
        Dict mapping relative file path to True (match) or False (mismatch/missing).

    Raises:
        FileNotFoundError: If the version or its manifest does not exist.
    """
    manifest = get_manifest(version_id)
    version_dir = PATHS.versions_dir / version_id
    results: Dict[str, bool] = {}

    for rel_path, expected_hash in manifest.checksums.items():
        file_path = version_dir / rel_path
        if not file_path.exists():
            results[rel_path] = False
            continue
        results[rel_path] = _sha256(file_path) == expected_hash

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _atomic_write_pointer(version_id: str) -> None:
    """Write version_id to active_version atomically via a sibling temp file."""
    pointer_path = PATHS.active_version_file
    pointer_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=pointer_path.parent,
        prefix=".tmp_active_version_",
        delete=False,
    ) as tmp:
        tmp.write(version_id)
        tmp_path = tmp.name

    os.replace(tmp_path, pointer_path)


def _write_manifest(version_dir: Path, manifest: VersionManifest) -> None:
    """Write a VersionManifest to manifest.json inside version_dir atomically."""
    manifest_path = version_dir / _MANIFEST_FILENAME

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=version_dir,
        prefix=".tmp_manifest_",
        suffix=".json",
        delete=False,
    ) as tmp:
        json.dump(manifest.to_dict(), tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, manifest_path)


def _read_manifest(version_dir: Path) -> VersionManifest:
    """Read and parse the manifest.json from a version directory."""
    with open(version_dir / _MANIFEST_FILENAME) as f:
        return VersionManifest.from_dict(json.load(f))


def _compute_checksums(version_dir: Path) -> Dict[str, str]:
    """
    Compute SHA-256 checksums for all artifact files in the versioned subdirectories.

    Returns a dict mapping paths relative to version_dir to their hex digest.
    Skips manifest.json (cannot contain its own checksum) and the data export
    sentinel file (a control file, not a data artifact).
    """
    checksums: Dict[str, str] = {}
    excluded = {_MANIFEST_FILENAME, f"data/{_EXPORT_SENTINEL}"}

    for subdir_name in _VERSIONED_SUBDIRS:
        subdir = version_dir / subdir_name
        if not subdir.exists():
            continue
        for file_path in sorted(subdir.rglob("*")):
            if file_path.is_dir():
                continue
            rel = file_path.relative_to(version_dir)
            if str(rel) in excluded:
                continue
            checksums[str(rel)] = _sha256(file_path)

    return checksums


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file in streaming chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHECKSUM_BLOCK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_metrics(source_dir: Path) -> Dict:
    """
    Read training_metrics.json from source_dir if present.

    Returns an empty dict if the file is absent or malformed, so a missing
    metrics file never blocks promotion.
    """
    metrics_path = source_dir / _METRICS_FILENAME
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _assert_staging_complete() -> None:
    """
    Raise RuntimeError if staging is missing any expected subdirectory or if
    the data export did not complete cleanly.

    Directory presence is a necessary but not sufficient condition for the
    data subdir — a crash mid-export leaves the directory populated but
    incomplete. The sentinel file written by export_training_data.py as its
    final step confirms the export finished without error.
    """
    missing = [sub for sub in _VERSIONED_SUBDIRS if not (PATHS.staging_dir / sub).exists()]
    if missing:
        raise RuntimeError(
            f"Staging is incomplete. Missing subdirectories: {missing}. "
            "Ensure all training scripts completed successfully before promoting."
        )

    sentinel = PATHS.staging_data_dir / _EXPORT_SENTINEL
    if not sentinel.exists():
        raise RuntimeError(
            f"Data export sentinel not found at '{sentinel}'. "
            "The export script did not complete successfully. "
            "Re-run export_training_data.py before promoting."
        )


def _git_short_hash(length: int = 6) -> Optional[str]:
    """Return the current git commit short hash, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"--short={length}", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _random_hex(length: int) -> str:
    """Return a random lowercase hex string of the given length."""
    return os.urandom(length // 2 + 1).hex()[:length]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_status() -> None:
    """Print the active version and a summary of its manifest."""
    try:
        version_id = PATHS.active_version_id()
        manifest = get_manifest(version_id)
    except FileNotFoundError as exc:
        print(f"No active version: {exc}")
        sys.exit(1)

    print(f"Active version : {manifest.version_id}")
    print(f"Created at     : {manifest.created_at}")
    print(f"Git commit     : {manifest.git_commit or 'unknown'}")
    print(f"PAD_IDX        : {manifest.pad_idx}")
    print(f"Attn strategy  : {manifest.attn_strategy}")

    als = manifest.metrics.get("scripts", {}).get("als", {})
    recall = als.get("metrics", {}).get("recall_at_30")
    if recall is not None:
        print(f"Recall@30 (ALS): {recall}")


def _cli_list() -> None:
    """Print all registered versions sorted newest first."""
    versions = list_versions()
    if not versions:
        print("No registered versions found.")
        return

    try:
        active_id = PATHS.active_version_id()
    except (FileNotFoundError, ValueError):
        active_id = None

    for manifest in versions:
        marker = " (active)" if manifest.version_id == active_id else ""
        als = manifest.metrics.get("scripts", {}).get("als", {})
        recall = als.get("metrics", {}).get("recall_at_30")
        recall_str = f"  recall@30={recall}" if recall is not None else ""
        print(f"{manifest.version_id}{marker}  created={manifest.created_at}{recall_str}")


def _cli_rollback(version_id: str) -> None:
    """Roll back to the given version ID and signal containers to reload."""
    from ops.training.reload_signal import signal_workers_reload

    try:
        rollback(version_id)
    except FileNotFoundError as exc:
        print(f"Rollback failed: {exc}")
        sys.exit(1)

    print("Signalling model server containers to reload...")
    try:
        signal_workers_reload()
    except RuntimeError as exc:
        print(f"Warning: worker reload signal failed: {exc}")
        print("Active version pointer has been updated. Restart containers manually to apply.")


def _cli_retire(keep: int) -> None:
    """Retire old versions, keeping the most recent N."""
    retired = retire_old_versions(keep=keep)
    if retired:
        print(f"Retired {len(retired)} version(s): {retired}")
    else:
        print("Nothing to retire.")


def _cli_verify(version_id: str) -> None:
    """Verify checksums for the given version and report any mismatches."""
    try:
        results = verify_checksums(version_id)
    except FileNotFoundError as exc:
        print(f"Verification failed: {exc}")
        sys.exit(1)

    failures = [path for path, ok in results.items() if not ok]
    if failures:
        print(f"CHECKSUM FAILURES in version '{version_id}':")
        for path in failures:
            print(f"  FAIL  {path}")
        sys.exit(1)
    else:
        print(f"All {len(results)} checksums verified for version '{version_id}'.")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artifact_registry",
        description="Manage versioned model artifact lifecycle.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show active version details.")
    sub.add_parser("list", help="List all registered versions.")

    rollback_p = sub.add_parser("rollback", help="Roll back to a previous version.")
    rollback_p.add_argument("--to", required=True, metavar="VERSION_ID", dest="version_id")

    retire_p = sub.add_parser("retire", help="Delete old versions beyond the keep limit.")
    retire_p.add_argument("--keep", type=int, default=5, metavar="N")

    verify_p = sub.add_parser("verify", help="Verify checksums for a version.")
    verify_p.add_argument("version_id", metavar="VERSION_ID")

    return parser


if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "status":
        _cli_status()
    elif args.command == "list":
        _cli_list()
    elif args.command == "rollback":
        _cli_rollback(args.version_id)
    elif args.command == "retire":
        _cli_retire(args.keep)
    elif args.command == "verify":
        _cli_verify(args.version_id)
