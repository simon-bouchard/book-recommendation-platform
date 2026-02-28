# tests/unit/ops/training/test_evaluate_gate.py
"""
Unit tests for the deployment quality gate.

All tests use tmp_path to isolate filesystem state. PATHS attributes are
monkeypatched to point at temporary directories so no real artifacts are
read or written during the test run.
"""

import json
import pytest
from pathlib import Path
from typing import Optional

from ops.training.evaluate_gate import (
    PromotionDecision,
    _DEFAULT_RECALL_FLOOR,
    _DEFAULT_RECALL_MAX_REGRESSION,
    _extract_recall,
    evaluate,
)
from models.core import paths as paths_module
from models.core.paths import PATHS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_staging_metrics(staging_dir: Path, recall: Optional[float]) -> None:
    """
    Write a training_metrics.json into staging_dir.

    If recall is None, writes a document with no ALS entry to simulate a
    failed or incomplete training run.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    if recall is None:
        doc = {"scripts": {}}
    else:
        doc = {
            "scripts": {
                "als": {
                    "metrics": {
                        "recall_at_30": recall,
                        "n_users": 1000,
                        "n_items": 50000,
                    },
                    "recorded_at": "2026-02-27T06:00:00+00:00",
                }
            }
        }

    with open(staging_dir / "training_metrics.json", "w") as f:
        json.dump(doc, f)


def _write_active_version(
    versions_dir: Path,
    active_version_file: Path,
    version_id: str,
    recall: Optional[float],
) -> None:
    """
    Write a versioned manifest and update the active_version pointer.

    The manifest embeds a training_metrics.json document under the `metrics`
    key, matching the structure that register_existing_version produces.
    """
    version_dir = versions_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    if recall is None:
        metrics = {"scripts": {}}
    else:
        metrics = {
            "scripts": {
                "als": {
                    "metrics": {"recall_at_30": recall},
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                }
            }
        }

    manifest = {
        "version_id": version_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "pad_idx": 0,
        "attn_strategy": "scalar",
        "metrics": metrics,
        "checksums": {},
    }

    with open(version_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    active_version_file.parent.mkdir(parents=True, exist_ok=True)
    active_version_file.write_text(version_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """
    Redirect all PATHS attributes used by evaluate_gate to tmp_path.

    Runs automatically for every test in this module so no test can
    accidentally read from or write to real artifact directories.
    """
    staging_dir = tmp_path / "staging"
    versions_dir = tmp_path / "versions"
    active_version_file = tmp_path / "active_version"

    monkeypatch.setattr(PATHS, "staging_dir", staging_dir)
    monkeypatch.setattr(PATHS, "versions_dir", versions_dir)
    monkeypatch.setattr(PATHS, "active_version_file", active_version_file)

    return tmp_path


# ---------------------------------------------------------------------------
# First deployment — no active version
# ---------------------------------------------------------------------------


class TestFirstDeployment:
    """Gate behaviour when no active version exists yet."""

    def test_approves_unconditionally_when_no_active_version(self, isolated_paths):
        """With no active_version pointer, the gate should approve without checking metrics."""
        decision = evaluate()

        assert decision.approved is True
        assert "First deployment" in decision.reason
        assert decision.active_recall is None

    def test_first_deployment_approval_carries_staging_recall_if_present(self, isolated_paths):
        """Staging recall should be surfaced in the decision even on first deployment."""
        _write_staging_metrics(PATHS.staging_dir, recall=0.42)

        decision = evaluate()

        assert decision.approved is True
        assert decision.staging_recall == pytest.approx(0.42)

    def test_first_deployment_approval_when_staging_metrics_absent(self, isolated_paths):
        """Missing staging metrics on first deployment should still approve."""
        decision = evaluate()

        assert decision.approved is True
        assert decision.staging_recall is None


# ---------------------------------------------------------------------------
# Missing staging metrics
# ---------------------------------------------------------------------------


class TestMissingStagingMetrics:
    """Gate behaviour when staging metrics are absent or incomplete."""

    def test_rejects_when_staging_metrics_file_absent(self, isolated_paths, tmp_path):
        """No training_metrics.json in staging should result in rejection."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)

        decision = evaluate()

        assert decision.approved is False
        assert "recall_at_30" in decision.reason

    def test_rejects_when_als_entry_missing_from_metrics(self, isolated_paths):
        """A metrics file with no ALS section should be rejected."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=None)

        decision = evaluate()

        assert decision.approved is False
        assert decision.staging_recall is None

    def test_rejects_when_staging_metrics_is_malformed_json(self, isolated_paths):
        """Malformed JSON in staging metrics should propagate as a decode error."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        PATHS.staging_dir.mkdir(parents=True, exist_ok=True)
        (PATHS.staging_dir / "training_metrics.json").write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            evaluate()


# ---------------------------------------------------------------------------
# Hard floor check
# ---------------------------------------------------------------------------


class TestHardFloor:
    """Gate behaviour for the absolute Recall@30 minimum."""

    def test_rejects_when_recall_below_default_floor(self, isolated_paths):
        """Staging recall below 0.25 should be rejected regardless of active version."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=0.10)

        decision = evaluate()

        assert decision.approved is False
        assert "hard floor" in decision.reason
        assert decision.staging_recall == pytest.approx(0.10)

    def test_rejects_exactly_at_floor_boundary(self, isolated_paths):
        """Recall exactly equal to the floor should be rejected (strict less-than check)."""
        floor = _DEFAULT_RECALL_FLOOR
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=floor)

        decision = evaluate()

        assert decision.approved is False

    def test_approves_when_recall_just_above_floor(self, isolated_paths):
        """Recall just above the floor should pass the floor check."""
        floor = _DEFAULT_RECALL_FLOOR
        staging_recall = floor + 0.001
        _write_active_version(
            PATHS.versions_dir, PATHS.active_version_file, "v1", recall=staging_recall
        )
        _write_staging_metrics(PATHS.staging_dir, recall=staging_recall)

        decision = evaluate()

        assert decision.approved is True

    def test_custom_floor_via_argument(self, isolated_paths):
        """A custom recall_floor argument should override the default."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=0.30)

        decision = evaluate(recall_floor=0.35)

        assert decision.approved is False
        assert "hard floor" in decision.reason

    def test_custom_floor_via_env_var(self, isolated_paths, monkeypatch):
        """GATE_RECALL_FLOOR env var should override the default floor."""
        monkeypatch.setenv("GATE_RECALL_FLOOR", "0.35")
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=0.30)

        decision = evaluate()

        assert decision.approved is False
        assert "hard floor" in decision.reason


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


class TestRegressionCheck:
    """Gate behaviour for the Recall@30 regression delta against the active version."""

    def test_rejects_when_regression_exceeds_max(self, isolated_paths):
        """A drop greater than 0.03 from the active version should be rejected."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.50)
        _write_staging_metrics(PATHS.staging_dir, recall=0.46)

        decision = evaluate()

        assert decision.approved is False
        assert "regression" in decision.reason.lower() or "dropped" in decision.reason.lower()
        assert decision.active_recall == pytest.approx(0.50)
        assert decision.staging_recall == pytest.approx(0.46)

    def test_rejects_exactly_at_regression_boundary(self, isolated_paths):
        """A drop exactly equal to max_regression should be rejected (strict greater-than)."""
        active_recall = 0.50
        max_delta = _DEFAULT_RECALL_MAX_REGRESSION
        _write_active_version(
            PATHS.versions_dir, PATHS.active_version_file, "v1", recall=active_recall
        )
        _write_staging_metrics(PATHS.staging_dir, recall=active_recall - max_delta)

        decision = evaluate()

        assert decision.approved is False

    def test_approves_when_regression_within_tolerance(self, isolated_paths):
        """A drop smaller than max_regression should be approved."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.50)
        _write_staging_metrics(PATHS.staging_dir, recall=0.48)

        decision = evaluate()

        assert decision.approved is True

    def test_approves_when_recall_improves(self, isolated_paths):
        """An improvement in recall should always be approved."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=0.55)

        decision = evaluate()

        assert decision.approved is True

    def test_approves_when_active_version_has_no_recall_metric(self, isolated_paths):
        """
        If the active manifest has no recall metric (e.g. legacy migration version),
        only the floor check applies — regression check is skipped.
        """
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=None)
        _write_staging_metrics(PATHS.staging_dir, recall=0.40)

        decision = evaluate()

        assert decision.approved is True
        assert decision.active_recall is None

    def test_custom_max_regression_via_argument(self, isolated_paths):
        """A custom recall_max_regression argument should override the default."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.50)
        _write_staging_metrics(PATHS.staging_dir, recall=0.48)

        decision = evaluate(recall_max_regression=0.01)

        assert decision.approved is False

    def test_custom_max_regression_via_env_var(self, isolated_paths, monkeypatch):
        """GATE_RECALL_MAX_REGRESSION env var should override the default."""
        monkeypatch.setenv("GATE_RECALL_MAX_REGRESSION", "0.01")
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.50)
        _write_staging_metrics(PATHS.staging_dir, recall=0.48)

        decision = evaluate()

        assert decision.approved is False


# ---------------------------------------------------------------------------
# Approval — full passing case
# ---------------------------------------------------------------------------


class TestApproval:
    """Gate behaviour when all checks pass."""

    def test_approved_decision_carries_both_recall_values(self, isolated_paths):
        """An approved decision should expose both staging and active recall."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=0.42)

        decision = evaluate()

        assert decision.approved is True
        assert decision.staging_recall == pytest.approx(0.42)
        assert decision.active_recall == pytest.approx(0.40)

    def test_approved_reason_contains_recall_values(self, isolated_paths):
        """The reason string should mention the staging recall value on approval."""
        _write_active_version(PATHS.versions_dir, PATHS.active_version_file, "v1", recall=0.40)
        _write_staging_metrics(PATHS.staging_dir, recall=0.42)

        decision = evaluate()

        assert "0.4200" in decision.reason


# ---------------------------------------------------------------------------
# _extract_recall unit tests
# ---------------------------------------------------------------------------


class TestExtractRecall:
    """Unit tests for the _extract_recall helper in isolation."""

    def test_extracts_recall_from_valid_document(self):
        """Should return the recall value from a well-formed metrics document."""
        doc = {"scripts": {"als": {"metrics": {"recall_at_30": 0.38}}}}
        assert _extract_recall(doc) == pytest.approx(0.38)

    def test_returns_none_when_scripts_key_missing(self):
        """Should return None if the top-level scripts key is absent."""
        assert _extract_recall({}) is None

    def test_returns_none_when_als_key_missing(self):
        """Should return None if the als script entry is absent."""
        assert _extract_recall({"scripts": {}}) is None

    def test_returns_none_when_metrics_key_missing(self):
        """Should return None if the metrics sub-key is absent."""
        assert _extract_recall({"scripts": {"als": {}}}) is None

    def test_returns_none_when_recall_key_missing(self):
        """Should return None if recall_at_30 is not in the metrics dict."""
        assert _extract_recall({"scripts": {"als": {"metrics": {}}}}) is None

    def test_returns_none_when_recall_value_is_none(self):
        """Should return None if recall_at_30 is explicitly null."""
        doc = {"scripts": {"als": {"metrics": {"recall_at_30": None}}}}
        assert _extract_recall(doc) is None

    def test_coerces_string_recall_to_float(self):
        """Should coerce a string recall value to float."""
        doc = {"scripts": {"als": {"metrics": {"recall_at_30": "0.42"}}}}
        assert _extract_recall(doc) == pytest.approx(0.42)

    def test_returns_none_for_non_numeric_string(self):
        """Should return None if the recall value cannot be converted to float."""
        doc = {"scripts": {"als": {"metrics": {"recall_at_30": "not_a_number"}}}}
        assert _extract_recall(doc) is None
