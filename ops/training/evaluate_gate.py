# ops/training/evaluate_gate.py
"""
Deployment quality gate for trained model artifacts.

Reads staging metrics and the active version manifest, then returns a typed
promotion decision. The gate is the only code that decides whether a training
run is safe to promote to production.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from models.core.artifact_registry import VersionManifest, get_manifest
from models.core.paths import PATHS

_METRICS_FILENAME = "training_metrics.json"

_DEFAULT_RECALL_FLOOR = 0.25
_DEFAULT_RECALL_MAX_REGRESSION = 0.03


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PromotionDecision:
    """
    Typed result of the deployment gate evaluation.

    Callers inspect `approved` to decide whether to proceed with promotion.
    `reason` is always populated and is suitable for logging and Slack messages.
    `staging_recall` and `active_recall` are included for observability —
    both may be None if the relevant metric was absent.
    """

    approved: bool
    reason: str
    staging_recall: Optional[float]
    active_recall: Optional[float]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def evaluate(
    recall_floor: float = _DEFAULT_RECALL_FLOOR,
    recall_max_regression: float = _DEFAULT_RECALL_MAX_REGRESSION,
) -> PromotionDecision:
    """
    Evaluate whether staging artifacts are safe to promote.

    Gate logic:
        1. If no active version exists (first-ever deployment), approve
           unconditionally — there is nothing to regress against.
        2. Hard floor: staging Recall@30 must exceed `recall_floor`.
           Catches catastrophic failures such as corrupt factors or broken
           ID mappings.
        3. Regression check: staging Recall@30 must not drop more than
           `recall_max_regression` below the active version's Recall@30.
           Catches genuine quality regressions from data issues.

    Subject embedding metrics recorded in training_metrics.json are embedded
    in the manifest for longitudinal tracking but are not part of the hard gate.

    Args:
        recall_floor: Absolute minimum Recall@30 required for promotion.
                      Configurable via the GATE_RECALL_FLOOR environment variable,
                      which takes precedence over this argument.
        recall_max_regression: Maximum allowed drop in Recall@30 versus the
                               active version. Configurable via
                               GATE_RECALL_MAX_REGRESSION.

    Returns:
        A PromotionDecision with approved, reason, and both recall values.

    Raises:
        FileNotFoundError: If staging/training_metrics.json does not exist.
        KeyError: If the ALS script entry is present but malformed.
    """
    recall_floor = float(os.getenv("GATE_RECALL_FLOOR", str(recall_floor)))
    recall_max_regression = float(
        os.getenv("GATE_RECALL_MAX_REGRESSION", str(recall_max_regression))
    )

    staging_recall = _read_staging_recall()
    active_manifest = _read_active_manifest()

    if active_manifest is None:
        return PromotionDecision(
            approved=True,
            reason="No active version found. First deployment — gate skipped.",
            staging_recall=staging_recall,
            active_recall=None,
        )

    if staging_recall is None:
        return PromotionDecision(
            approved=False,
            reason=(
                "Staging training_metrics.json is missing the ALS recall_at_30 metric. "
                "Ensure train_als.py completed successfully."
            ),
            staging_recall=None,
            active_recall=None,
        )

    active_recall = _extract_recall(active_manifest.metrics)

    if staging_recall < recall_floor:
        return PromotionDecision(
            approved=False,
            reason=(
                f"Recall@30 {staging_recall:.4f} is below the hard floor of "
                f"{recall_floor:.4f}. This indicates a catastrophic failure "
                f"(corrupt factors or broken ID mapping). Promotion rejected."
            ),
            staging_recall=staging_recall,
            active_recall=active_recall,
        )

    if active_recall is not None:
        regression = active_recall - staging_recall
        if regression > recall_max_regression:
            return PromotionDecision(
                approved=False,
                reason=(
                    f"Recall@30 dropped from {active_recall:.4f} to {staging_recall:.4f} "
                    f"(delta={regression:.4f}), exceeding the maximum allowed regression "
                    f"of {recall_max_regression:.4f}. Promotion rejected."
                ),
                staging_recall=staging_recall,
                active_recall=active_recall,
            )

    return PromotionDecision(
        approved=True,
        reason=(
            f"Gate passed. Recall@30={staging_recall:.4f}"
            + (f" (active={active_recall:.4f})" if active_recall is not None else "")
            + "."
        ),
        staging_recall=staging_recall,
        active_recall=active_recall,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_staging_recall() -> Optional[float]:
    """
    Read Recall@30 from staging/training_metrics.json.

    Returns None if the file is absent, the ALS entry is missing, or the
    recall key is not present — all treated as a missing metric by the gate.

    Raises:
        json.JSONDecodeError: If the file exists but is not valid JSON.
    """
    metrics_path = PATHS.staging_dir / _METRICS_FILENAME
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        doc = json.load(f)

    return _extract_recall(doc)


def _extract_recall(metrics_doc: dict) -> Optional[float]:
    """
    Extract recall_at_30 from a parsed training_metrics.json document.

    The document structure is:
        {"scripts": {"als": {"metrics": {"recall_at_30": float, ...}, ...}, ...}}

    Returns None if any level of the path is absent.
    """
    try:
        value = metrics_doc["scripts"]["als"]["metrics"]["recall_at_30"]
        return float(value)
    except (KeyError, TypeError, ValueError):
        return None


def _read_active_manifest() -> Optional[VersionManifest]:
    """
    Read the active version manifest.

    Returns None if no active version exists (first deployment), so the
    gate can approve unconditionally in that case.

    Raises:
        json.JSONDecodeError: If the manifest exists but is not valid JSON.
    """
    try:
        active_id = PATHS.active_version_id()
        return get_manifest(active_id)
    except FileNotFoundError:
        return None
