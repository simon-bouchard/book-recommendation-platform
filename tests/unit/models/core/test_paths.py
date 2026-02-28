# tests/unit/models/core/test_paths.py
"""
Unit tests for ModelPaths in models.core.paths.

Each test constructs a fresh ModelPaths(project_root=tmp_path) instance so
the global PATHS singleton is never touched and there is no shared state
between tests.
"""

import pytest
from pathlib import Path

from models.core.paths import ModelPaths, _VERSIONED_SUBDIRS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def paths(tmp_path) -> ModelPaths:
    """A ModelPaths instance rooted at a clean temporary directory."""
    return ModelPaths(project_root=tmp_path)


@pytest.fixture()
def paths_with_active(tmp_path) -> ModelPaths:
    """
    A ModelPaths instance with a valid active_version pointer already written.

    The pointer file is written before the instance is created so that
    properties depending on _active_version_dir work immediately.
    """
    p = ModelPaths(project_root=tmp_path)
    p.active_version_file.parent.mkdir(parents=True, exist_ok=True)
    p.active_version_file.write_text("20260227-1200-abc123")
    return p


# ---------------------------------------------------------------------------
# Constructor and static path layout
# ---------------------------------------------------------------------------


class TestConstructor:
    """Paths are derived correctly from project_root."""

    def test_models_root_is_under_project_root(self, paths, tmp_path):
        assert paths.models_root == tmp_path / "models"

    def test_artifacts_dir_is_under_models_root(self, paths):
        assert paths.artifacts_dir == paths.models_root / "artifacts"

    def test_staging_dir_is_under_artifacts(self, paths):
        assert paths.staging_dir == paths.artifacts_dir / "staging"

    def test_versions_dir_is_under_artifacts(self, paths):
        assert paths.versions_dir == paths.artifacts_dir / "versions"

    def test_active_version_file_is_under_artifacts(self, paths):
        assert paths.active_version_file == paths.artifacts_dir / "active_version"

    def test_semantic_indexes_dir_is_under_artifacts(self, paths):
        assert paths.semantic_indexes_dir == paths.artifacts_dir / "semantic_indexes"

    def test_training_data_dir_is_under_training_root(self, paths):
        assert paths.training_data_dir == paths.models_root / "training" / "data"


# ---------------------------------------------------------------------------
# _active_version_dir resolution
# ---------------------------------------------------------------------------


class TestActiveVersionDir:
    """Version pointer file read and error behaviour."""

    def test_raises_file_not_found_when_pointer_absent(self, paths):
        with pytest.raises(FileNotFoundError, match="active version pointer"):
            _ = paths._active_version_dir

    def test_raises_value_error_when_pointer_is_empty(self, paths):
        paths.active_version_file.parent.mkdir(parents=True, exist_ok=True)
        paths.active_version_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            _ = paths._active_version_dir

    def test_raises_value_error_when_pointer_is_only_whitespace(self, paths):
        paths.active_version_file.parent.mkdir(parents=True, exist_ok=True)
        paths.active_version_file.write_text("   \n  ")

        with pytest.raises(ValueError, match="empty"):
            _ = paths._active_version_dir

    def test_resolves_to_correct_version_dir(self, paths_with_active):
        result = paths_with_active._active_version_dir

        assert result == paths_with_active.versions_dir / "20260227-1200-abc123"

    def test_re_reads_pointer_on_every_access(self, paths):
        """Changing the pointer file should be reflected immediately."""
        paths.active_version_file.parent.mkdir(parents=True, exist_ok=True)
        paths.active_version_file.write_text("version-a")
        assert paths._active_version_dir.name == "version-a"

        paths.active_version_file.write_text("version-b")
        assert paths._active_version_dir.name == "version-b"

    def test_strips_trailing_newline_from_pointer(self, paths):
        paths.active_version_file.parent.mkdir(parents=True, exist_ok=True)
        paths.active_version_file.write_text("20260227-1200-abc123\n")

        assert paths._active_version_dir.name == "20260227-1200-abc123"


# ---------------------------------------------------------------------------
# active_version_id
# ---------------------------------------------------------------------------


class TestActiveVersionId:
    """Public accessor for the current version ID string."""

    def test_returns_version_id_string(self, paths_with_active):
        assert paths_with_active.active_version_id() == "20260227-1200-abc123"

    def test_raises_file_not_found_when_pointer_absent(self, paths):
        with pytest.raises(FileNotFoundError):
            paths.active_version_id()

    def test_raises_value_error_when_pointer_empty(self, paths):
        paths.active_version_file.parent.mkdir(parents=True, exist_ok=True)
        paths.active_version_file.write_text("")

        with pytest.raises(ValueError):
            paths.active_version_id()


# ---------------------------------------------------------------------------
# Versioned artifact directory properties
# ---------------------------------------------------------------------------


class TestVersionedDirectories:
    """embeddings_dir, attention_dir, and scoring_dir all delegate through the pointer."""

    def test_embeddings_dir_is_under_active_version(self, paths_with_active):
        expected = paths_with_active.versions_dir / "20260227-1200-abc123" / "embeddings"
        assert paths_with_active.embeddings_dir == expected

    def test_attention_dir_is_under_active_version(self, paths_with_active):
        expected = paths_with_active.versions_dir / "20260227-1200-abc123" / "attention"
        assert paths_with_active.attention_dir == expected

    def test_scoring_dir_is_under_active_version(self, paths_with_active):
        expected = paths_with_active.versions_dir / "20260227-1200-abc123" / "scoring"
        assert paths_with_active.scoring_dir == expected

    def test_all_versioned_subdirs_are_represented(self, paths_with_active):
        """Every name in _VERSIONED_SUBDIRS should have a corresponding property."""
        for subdir in _VERSIONED_SUBDIRS:
            prop_path = getattr(paths_with_active, f"{subdir}_dir")
            assert prop_path.parent.name == "20260227-1200-abc123"
            assert prop_path.name == subdir

    def test_versioned_dirs_raise_when_pointer_absent(self, paths):
        for subdir in _VERSIONED_SUBDIRS:
            with pytest.raises(FileNotFoundError):
                _ = getattr(paths, f"{subdir}_dir")

    def test_versioned_dirs_update_after_pointer_change(self, paths):
        """Changing the pointer should immediately change the resolved dir."""
        paths.active_version_file.parent.mkdir(parents=True, exist_ok=True)
        paths.active_version_file.write_text("v1")
        assert paths.embeddings_dir.parts[-2] == "v1"

        paths.active_version_file.write_text("v2")
        assert paths.embeddings_dir.parts[-2] == "v2"


# ---------------------------------------------------------------------------
# Versioned artifact file paths
# ---------------------------------------------------------------------------


class TestVersionedFilePaths:
    """Individual artifact file paths sit inside the correct versioned subdirs."""

    def test_book_subject_embeddings_is_under_embeddings_dir(self, paths_with_active):
        assert paths_with_active.book_subject_embeddings.parent == paths_with_active.embeddings_dir

    def test_book_subject_ids_is_under_embeddings_dir(self, paths_with_active):
        assert paths_with_active.book_subject_ids.parent == paths_with_active.embeddings_dir

    def test_book_als_factors_is_under_embeddings_dir(self, paths_with_active):
        assert paths_with_active.book_als_factors.parent == paths_with_active.embeddings_dir

    def test_user_als_factors_is_under_embeddings_dir(self, paths_with_active):
        assert paths_with_active.user_als_factors.parent == paths_with_active.embeddings_dir

    def test_bayesian_scores_is_under_scoring_dir(self, paths_with_active):
        assert paths_with_active.bayesian_scores.parent == paths_with_active.scoring_dir

    def test_attention_files_are_under_attention_dir(self, paths_with_active):
        for strategy in ("scalar", "perdim", "selfattn", "selfattn_perdim"):
            attr = f"subject_attention_{strategy}"
            path = getattr(paths_with_active, attr)
            assert path.parent == paths_with_active.attention_dir


# ---------------------------------------------------------------------------
# get_attention_path
# ---------------------------------------------------------------------------


class TestGetAttentionPath:
    """Strategy name dispatch and validation."""

    @pytest.mark.parametrize("strategy", ["scalar", "perdim", "selfattn", "selfattn_perdim"])
    def test_valid_strategy_returns_path_under_attention_dir(self, paths_with_active, strategy):
        path = paths_with_active.get_attention_path(strategy)
        assert path.parent == paths_with_active.attention_dir

    @pytest.mark.parametrize("strategy", ["scalar", "perdim", "selfattn", "selfattn_perdim"])
    def test_filename_contains_strategy_name(self, paths_with_active, strategy):
        path = paths_with_active.get_attention_path(strategy)
        assert strategy in path.name

    def test_invalid_strategy_raises_value_error(self, paths_with_active):
        with pytest.raises(ValueError, match="Unknown attention strategy"):
            paths_with_active.get_attention_path("invalid")

    def test_get_attention_path_matches_property(self, paths_with_active):
        """get_attention_path('scalar') and subject_attention_scalar should agree."""
        assert (
            paths_with_active.get_attention_path("scalar")
            == paths_with_active.subject_attention_scalar
        )


# ---------------------------------------------------------------------------
# Static (non-versioned) paths
# ---------------------------------------------------------------------------


class TestStaticPaths:
    """Paths that do not go through the active_version pointer."""

    def test_semantic_index_paths_do_not_require_active_version(self, paths):
        """Static index paths are accessible even without a pointer file."""
        _ = paths.semantic_index_baseline
        _ = paths.semantic_index_enriched_v1
        _ = paths.semantic_index_enriched_v2

    def test_training_data_paths_do_not_require_active_version(self, paths):
        _ = paths.training_interactions
        _ = paths.training_users
        _ = paths.training_books
        _ = paths.training_book_subjects

    def test_semantic_index_baseline_is_under_semantic_indexes_dir(self, paths):
        assert paths.semantic_index_baseline.parent == paths.semantic_indexes_dir

    def test_training_interactions_is_under_training_data_dir(self, paths):
        assert paths.training_interactions.parent == paths.training_data_dir


# ---------------------------------------------------------------------------
# Directory creation helpers
# ---------------------------------------------------------------------------


class TestEnsureDirs:
    """ensure_* helpers create the correct directories."""

    def test_ensure_artifact_dirs_creates_staging(self, paths):
        paths.ensure_artifact_dirs()
        assert paths.staging_dir.is_dir()

    def test_ensure_artifact_dirs_creates_versions(self, paths):
        paths.ensure_artifact_dirs()
        assert paths.versions_dir.is_dir()

    def test_ensure_artifact_dirs_creates_semantic_indexes(self, paths):
        paths.ensure_artifact_dirs()
        assert paths.semantic_indexes_dir.is_dir()

    def test_ensure_artifact_dirs_is_idempotent(self, paths):
        paths.ensure_artifact_dirs()
        paths.ensure_artifact_dirs()
        assert paths.staging_dir.is_dir()

    def test_ensure_staging_dirs_creates_all_versioned_subdirs(self, paths):
        paths.ensure_staging_dirs()
        for subdir in _VERSIONED_SUBDIRS:
            assert (paths.staging_dir / subdir).is_dir()

    def test_ensure_staging_dirs_is_idempotent(self, paths):
        paths.ensure_staging_dirs()
        paths.ensure_staging_dirs()
        for subdir in _VERSIONED_SUBDIRS:
            assert (paths.staging_dir / subdir).is_dir()

    def test_ensure_training_dirs_creates_training_data_dir(self, paths):
        paths.ensure_training_dirs()
        assert paths.training_data_dir.is_dir()
