# tests/unit/models/core/test_artifact_registry.py
"""
Unit tests for artifact lifecycle management in models.core.artifact_registry.

All tests use tmp_path to isolate filesystem state. PATHS instance attributes
and the module-level _RELOAD_SIGNAL_PATH constant are monkeypatched so no real
artifact directories are touched during the test run.
"""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

import models.core.artifact_registry as registry_module
from models.core.artifact_registry import (
    VersionManifest,
    generate_version_id,
    get_manifest,
    list_versions,
    promote_staging,
    register_existing_version,
    retire_old_versions,
    rollback,
    verify_checksums,
)
from models.core.paths import _VERSIONED_SUBDIRS, PATHS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_staging(staging_dir: Path, subdirs: tuple = _VERSIONED_SUBDIRS) -> None:
    """Create a staging directory with the expected subdirs and a dummy artifact in each."""
    for subdir in subdirs:
        d = staging_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{subdir}_data.npy").write_bytes(b"fake artifact content: " + subdir.encode())
    if "data" in subdirs:
        (staging_dir / "data" / ".export_complete").touch()


def _make_staging_with_metrics(staging_dir: Path, recall: float = 0.42) -> None:
    """Create a complete staging directory including training_metrics.json."""
    _make_staging(staging_dir)
    metrics = {
        "scripts": {
            "als": {
                "metrics": {"recall_at_30": recall},
                "recorded_at": "2026-02-27T06:00:00+00:00",
            }
        }
    }
    with open(staging_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f)


def _register(
    versions_dir: Path,
    active_version_file: Path,
    version_id: str,
    recall: Optional[float] = 0.40,
) -> VersionManifest:
    """
    Convenience wrapper to register a version directly from a temp source dir.

    Creates a minimal source directory with the required subdirs, then calls
    register_existing_version so the real promotion path is exercised.
    """
    source = versions_dir.parent / f"_src_{version_id}"
    _make_staging(source)
    if recall is not None:
        metrics = {"scripts": {"als": {"metrics": {"recall_at_30": recall}}}}
    else:
        metrics = {}
    return register_existing_version(source, version_id, metrics=metrics)


def _stamp(versions_dir: Path, version_id: str, created_at: str) -> None:
    """Overwrite the created_at field in a manifest to a controlled timestamp."""
    manifest_path = versions_dir / version_id / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["created_at"] = created_at
    manifest_path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """
    Redirect all PATHS attributes to tmp_path.

    autouse=True means every test in this module gets isolation automatically.
    """
    staging_dir = tmp_path / "staging"
    versions_dir = tmp_path / "versions"
    active_version_file = tmp_path / "active_version"

    monkeypatch.setattr(PATHS, "staging_dir", staging_dir)
    monkeypatch.setattr(PATHS, "versions_dir", versions_dir)
    monkeypatch.setattr(PATHS, "active_version_file", active_version_file)

    return tmp_path


# ---------------------------------------------------------------------------
# generate_version_id
# ---------------------------------------------------------------------------


class TestGenerateVersionId:
    """Version ID format and uniqueness."""

    def test_returns_string(self):
        assert isinstance(generate_version_id(), str)

    def test_format_has_three_parts(self):
        """Should be YYYYMMDD-HHMM-{suffix}."""
        parts = generate_version_id().split("-")
        assert len(parts) == 3

    def test_date_part_is_8_digits(self):
        date_part = generate_version_id().split("-")[0]
        assert date_part.isdigit() and len(date_part) == 8

    def test_time_part_is_4_digits(self):
        time_part = generate_version_id().split("-")[1]
        assert time_part.isdigit() and len(time_part) == 4

    def test_suffix_is_6_chars(self):
        suffix = generate_version_id().split("-")[2]
        assert len(suffix) == 6

    def test_two_calls_produce_different_ids(self):
        """IDs should be unique across calls (git hash or random suffix differs)."""
        with patch.object(registry_module, "_git_short_hash", return_value=None):
            id1 = generate_version_id()
            id2 = generate_version_id()
        assert id1 != id2


# ---------------------------------------------------------------------------
# register_existing_version
# ---------------------------------------------------------------------------


class TestRegisterExistingVersion:
    """Core promotion primitive."""

    def test_creates_version_directory(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)

        register_existing_version(source, "v1")

        assert (PATHS.versions_dir / "v1").is_dir()

    def test_copies_all_subdirs(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)

        register_existing_version(source, "v1")

        for subdir in _VERSIONED_SUBDIRS:
            assert (PATHS.versions_dir / "v1" / subdir).is_dir()

    def test_writes_manifest_json(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)

        register_existing_version(source, "v1")

        manifest_path = PATHS.versions_dir / "v1" / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_contains_expected_fields(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)

        manifest = register_existing_version(source, "v1")

        assert manifest.version_id == "v1"
        assert manifest.created_at
        assert isinstance(manifest.checksums, dict)

    def test_manifest_embeds_provided_metrics(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        metrics = {"scripts": {"als": {"metrics": {"recall_at_30": 0.45}}}}

        manifest = register_existing_version(source, "v1", metrics=metrics)

        assert manifest.metrics == metrics

    def test_manifest_reads_metrics_from_source_if_not_provided(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        metrics = {"scripts": {"als": {"metrics": {"recall_at_30": 0.39}}}}
        with open(source / "training_metrics.json", "w") as f:
            json.dump(metrics, f)

        manifest = register_existing_version(source, "v1")

        assert manifest.metrics == metrics

    def test_checksums_are_computed_for_all_artifact_files(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)

        manifest = register_existing_version(source, "v1")

        assert len(manifest.checksums) == len(_VERSIONED_SUBDIRS)
        for key in manifest.checksums:
            assert isinstance(manifest.checksums[key], str)
            assert len(manifest.checksums[key]) == 64  # SHA-256 hex

    def test_updates_active_version_pointer(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)

        register_existing_version(source, "v1")

        assert PATHS.active_version_file.read_text() == "v1"

    def test_raises_if_source_does_not_exist(self, isolated_registry):
        with pytest.raises(FileNotFoundError, match="Source directory"):
            register_existing_version(isolated_registry / "nonexistent", "v1")

    def test_raises_if_version_already_exists(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        register_existing_version(source, "v1")

        source2 = isolated_registry / "source2"
        _make_staging(source2)

        with pytest.raises(FileExistsError, match="already exists"):
            register_existing_version(source2, "v1")


# ---------------------------------------------------------------------------
# promote_staging
# ---------------------------------------------------------------------------


class TestPromoteStaging:
    """Staging-specific promotion wrapper."""

    def test_promotes_complete_staging(self, isolated_registry):
        _make_staging_with_metrics(PATHS.staging_dir)

        manifest = promote_staging("v1")

        assert manifest.version_id == "v1"
        assert (PATHS.versions_dir / "v1").is_dir()

    def test_active_version_set_after_promotion(self, isolated_registry):
        _make_staging_with_metrics(PATHS.staging_dir)

        promote_staging("v1")

        assert PATHS.active_version_file.read_text() == "v1"

    def test_raises_when_staging_missing_a_subdir(self, isolated_registry):
        """Staging with only some subdirs present should be rejected."""
        incomplete_subdirs = _VERSIONED_SUBDIRS[:-1]
        _make_staging(PATHS.staging_dir, subdirs=incomplete_subdirs)

        with pytest.raises(RuntimeError, match="incomplete"):
            promote_staging("v1")

    def test_raises_when_staging_is_empty(self, isolated_registry):
        PATHS.staging_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RuntimeError, match="incomplete"):
            promote_staging("v1")


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


class TestRollback:
    """Active version pointer switching."""

    def test_rollback_updates_active_pointer(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        _register(PATHS.versions_dir, PATHS.active_version_file, "v2")
        assert PATHS.active_version_file.read_text() == "v2"

        rollback("v1")

        assert PATHS.active_version_file.read_text() == "v1"

    def test_rollback_does_not_signal_workers(self, isolated_registry):
        """
        rollback() is pointer-only. Triggering a worker reload is the caller's
        responsibility (ops/training/reload_signal.signal_workers_reload).
        """
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        _register(PATHS.versions_dir, PATHS.active_version_file, "v2")

        with patch("subprocess.run") as mock_run:
            rollback("v1")

        mock_run.assert_not_called()

    def test_rollback_raises_for_unknown_version(self, isolated_registry):
        with pytest.raises(FileNotFoundError, match="not found"):
            rollback("nonexistent-version")

    def test_rollback_does_not_modify_artifact_files(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        _register(PATHS.versions_dir, PATHS.active_version_file, "v2")

        files_before = set(PATHS.versions_dir.rglob("*"))
        rollback("v1")
        files_after = set(PATHS.versions_dir.rglob("*"))

        assert files_before == files_after


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------


class TestListVersions:
    """Version enumeration and sorting."""

    def test_returns_empty_list_when_no_versions_dir(self, isolated_registry):
        assert list_versions() == []

    def test_returns_empty_list_when_versions_dir_is_empty(self, isolated_registry):
        PATHS.versions_dir.mkdir(parents=True)
        assert list_versions() == []

    def test_returns_all_registered_versions(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        _register(PATHS.versions_dir, PATHS.active_version_file, "v2")
        _register(PATHS.versions_dir, PATHS.active_version_file, "v3")

        versions = list_versions()

        assert len(versions) == 3
        assert {m.version_id for m in versions} == {"v1", "v2", "v3"}

    def test_versions_sorted_newest_first(self, isolated_registry):
        """Sorted by created_at descending — we stub the timestamp to control order."""
        timestamps = [
            "2026-01-01T00:00:00+00:00",
            "2026-02-01T00:00:00+00:00",
            "2026-03-01T00:00:00+00:00",
        ]
        for vid, ts in zip(["v1", "v2", "v3"], timestamps):
            source = isolated_registry / f"src_{vid}"
            _make_staging(source)
            register_existing_version(source, vid)
            # Overwrite the manifest with a controlled timestamp
            manifest_path = PATHS.versions_dir / vid / "manifest.json"
            data = json.loads(manifest_path.read_text())
            data["created_at"] = ts
            manifest_path.write_text(json.dumps(data))

        versions = list_versions()

        assert [m.version_id for m in versions] == ["v3", "v2", "v1"]

    def test_skips_version_dir_without_manifest(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        (PATHS.versions_dir / "orphan-dir").mkdir()

        versions = list_versions()

        assert len(versions) == 1
        assert versions[0].version_id == "v1"

    def test_skips_version_with_malformed_manifest(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        bad_dir = PATHS.versions_dir / "bad"
        bad_dir.mkdir()
        (bad_dir / "manifest.json").write_text("not json {{{")

        versions = list_versions()

        assert len(versions) == 1


# ---------------------------------------------------------------------------
# get_manifest
# ---------------------------------------------------------------------------


class TestGetManifest:
    """Single-version manifest retrieval."""

    def test_returns_manifest_for_existing_version(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")

        manifest = get_manifest("v1")

        assert isinstance(manifest, VersionManifest)
        assert manifest.version_id == "v1"

    def test_raises_for_nonexistent_version(self, isolated_registry):
        with pytest.raises(FileNotFoundError):
            get_manifest("does-not-exist")

    def test_manifest_roundtrip_preserves_fields(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        metrics = {"scripts": {"als": {"metrics": {"recall_at_30": 0.42}}}}
        original = register_existing_version(source, "v1", metrics=metrics)

        retrieved = get_manifest("v1")

        assert retrieved.version_id == original.version_id
        assert retrieved.metrics == original.metrics
        assert retrieved.checksums == original.checksums


# ---------------------------------------------------------------------------
# retire_old_versions
# ---------------------------------------------------------------------------


class TestRetireOldVersions:
    """Version pruning with protected active and predecessor."""

    def test_raises_when_keep_less_than_2(self, isolated_registry):
        with pytest.raises(ValueError, match="at least 2"):
            retire_old_versions(keep=1)

    def test_returns_empty_list_when_nothing_to_retire(self, isolated_registry):
        _register(PATHS.versions_dir, PATHS.active_version_file, "v1")
        _register(PATHS.versions_dir, PATHS.active_version_file, "v2")

        retired = retire_old_versions(keep=5)

        assert retired == []

    def test_retires_versions_beyond_keep_limit(self, isolated_registry):
        for i in range(1, 8):
            _register(PATHS.versions_dir, PATHS.active_version_file, f"v{i}")
            _stamp(PATHS.versions_dir, f"v{i}", f"2026-01-{i:02d}T00:00:00+00:00")

        retired = retire_old_versions(keep=5)

        remaining = list_versions()
        assert len(remaining) == 5
        assert len(retired) == 2

    def test_always_protects_active_version(self, isolated_registry):
        for i in range(1, 8):
            _register(PATHS.versions_dir, PATHS.active_version_file, f"v{i}")
            _stamp(PATHS.versions_dir, f"v{i}", f"2026-01-{i:02d}T00:00:00+00:00")
        active_id = PATHS.active_version_file.read_text()

        retire_old_versions(keep=2)

        remaining_ids = {m.version_id for m in list_versions()}
        assert active_id in remaining_ids

    def test_always_protects_predecessor_of_active(self, isolated_registry):
        """The version immediately before the active one must survive for fast rollback."""
        for i in range(1, 8):
            _register(PATHS.versions_dir, PATHS.active_version_file, f"v{i}")
            _stamp(PATHS.versions_dir, f"v{i}", f"2026-01-{i:02d}T00:00:00+00:00")

        # With deterministic timestamps, newest-first order is v7, v6, v5, ..., v1
        # Active is v7 (index 0), predecessor is v6 (index 1)
        predecessor_id = "v6"

        retire_old_versions(keep=2)

        remaining_ids = {m.version_id for m in list_versions()}
        assert predecessor_id in remaining_ids

    def test_removes_version_directories_from_disk(self, isolated_registry):
        for i in range(1, 5):
            _register(PATHS.versions_dir, PATHS.active_version_file, f"v{i}")
            _stamp(PATHS.versions_dir, f"v{i}", f"2026-01-{i:02d}T00:00:00+00:00")

        retired = retire_old_versions(keep=2)

        for vid in retired:
            assert not (PATHS.versions_dir / vid).exists()

    def test_returns_empty_when_no_versions_exist(self, isolated_registry):
        retired = retire_old_versions(keep=5)
        assert retired == []


# ---------------------------------------------------------------------------
# verify_checksums
# ---------------------------------------------------------------------------


class TestVerifyChecksums:
    """Checksum integrity verification."""

    def test_all_pass_for_unmodified_version(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        register_existing_version(source, "v1")

        results = verify_checksums("v1")

        assert all(results.values())
        assert len(results) == len(_VERSIONED_SUBDIRS)

    def test_detects_modified_file(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        register_existing_version(source, "v1")

        artifact_file = next(f for f in (PATHS.versions_dir / "v1").rglob("*.npy"))
        artifact_file.write_bytes(b"tampered content")

        results = verify_checksums("v1")

        rel = str(artifact_file.relative_to(PATHS.versions_dir / "v1"))
        assert results[rel] is False

    def test_detects_missing_file(self, isolated_registry):
        source = isolated_registry / "source"
        _make_staging(source)
        register_existing_version(source, "v1")

        artifact_file = next(f for f in (PATHS.versions_dir / "v1").rglob("*.npy"))
        rel = str(artifact_file.relative_to(PATHS.versions_dir / "v1"))
        artifact_file.unlink()

        results = verify_checksums("v1")

        assert results[rel] is False

    def test_raises_for_nonexistent_version(self, isolated_registry):
        with pytest.raises(FileNotFoundError):
            verify_checksums("does-not-exist")

    def test_returns_empty_dict_when_no_checksums_in_manifest(self, isolated_registry):
        """A manifest with an empty checksums dict produces an empty result."""
        source = isolated_registry / "source"
        _make_staging(source)
        register_existing_version(source, "v1")

        manifest_path = PATHS.versions_dir / "v1" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["checksums"] = {}
        manifest_path.write_text(json.dumps(data))

        results = verify_checksums("v1")

        assert results == {}


# ---------------------------------------------------------------------------
# VersionManifest serialisation
# ---------------------------------------------------------------------------


class TestVersionManifestSerialisation:
    """Round-trip serialisation for the VersionManifest dataclass."""

    def test_to_dict_and_from_dict_roundtrip(self):
        original = VersionManifest(
            version_id="20260227-1200-abc123",
            created_at="2026-02-27T12:00:00+00:00",
            git_commit="abc123",
            pad_idx=42,
            attn_strategy="scalar",
            metrics={"scripts": {"als": {"metrics": {"recall_at_30": 0.38}}}},
            checksums={"embeddings/data.npy": "deadbeef" * 8},
        )

        recovered = VersionManifest.from_dict(original.to_dict())

        assert recovered == original

    def test_from_dict_tolerates_missing_optional_fields(self):
        minimal = {
            "version_id": "v1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "pad_idx": 0,
            "attn_strategy": "scalar",
        }
        manifest = VersionManifest.from_dict(minimal)

        assert manifest.git_commit is None
        assert manifest.metrics == {}
        assert manifest.checksums == {}
