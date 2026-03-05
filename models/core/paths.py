# models/core/paths.py
"""
Centralized path definitions for all model artifacts and data files.
Versioned artifact directories (embeddings, attention, scoring, data) resolve
through the active_version pointer file, so loaders always read from the correct version.
"""

from pathlib import Path
from typing import Literal

AttentionStrategy = Literal["scalar", "perdim", "selfattn", "selfattn_perdim"]

_ACTIVE_VERSION_FILENAME = "active_version"
_VERSIONED_SUBDIRS = ("embeddings", "attention", "scoring", "data")


class ModelPaths:
    """
    Centralized registry of all model artifact paths.

    Versioned artifact directories are resolved dynamically through
    models/artifacts/active_version, a plain-text file containing a version ID
    such as '20260226-1430-a3f9b2'. All other paths are static.

    Directory structure:
        models/
        |-- artifacts/
        |   |-- active_version              # plain text: "20260226-1430-a3f9b2"
        |   |-- staging/                    # new artifacts land here first
        |   |   |-- training_metrics.json
        |   |   |-- embeddings/
        |   |   |-- attention/
        |   |   |-- scoring/
        |   |   `-- data/
        |   |-- versions/
        |   |   |-- 20260226-1430-a3f9b2/
        |   |   |   |-- manifest.json
        |   |   |   |-- embeddings/
        |   |   |   |-- attention/
        |   |   |   |-- scoring/
        |   |   |   `-- data/
        |   |   `-- 20260110-0900-c8d1e4/  # previous versions kept for rollback
        |   |       `-- ...
        |   `-- semantic_indexes/           # not versioned
        `-- training/
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize path registry.

        Args:
            project_root: Root directory of the project. Defaults to three levels
                          above this file (i.e. the repository root).
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent

        self.project_root = project_root
        self.models_root = project_root / "models"

        self.artifacts_dir = self.models_root / "artifacts"
        self.training_root = self.models_root / "training"

        self.staging_dir = self.artifacts_dir / "staging"
        self.versions_dir = self.artifacts_dir / "versions"
        self.active_version_file = self.artifacts_dir / _ACTIVE_VERSION_FILENAME

        self.semantic_indexes_dir = self.artifacts_dir / "semantic_indexes"

    # -------------------------------------------------------------------------
    # Internal version resolution
    # -------------------------------------------------------------------------

    @property
    def _active_version_dir(self) -> Path:
        """
        Resolve the active versioned artifact directory.

        Reads the active_version pointer file on every call so that a rollback
        takes effect on the next access without restarting the application.

        Raises:
            FileNotFoundError: If active_version does not exist, which means
                migration has not been run yet.
            ValueError: If active_version exists but is empty or blank.
        """
        if not self.active_version_file.exists():
            raise FileNotFoundError(
                f"No active version pointer found at '{self.active_version_file}'. "
                "Initialize the versioned artifact structure by running: "
                "python ops/migrations/migrate_flat_artifacts.py"
            )

        version_id = self.active_version_file.read_text().strip()

        if not version_id:
            raise ValueError(
                f"Active version pointer file is empty: '{self.active_version_file}'. "
                "The file must contain a valid version ID."
            )

        return self.versions_dir / version_id

    def active_version_id(self) -> str:
        """
        Return the currently active version ID string.

        Delegates to _active_version_dir so it raises the same errors on
        misconfiguration. Useful for logging and manifest lookups.
        """
        return self._active_version_dir.name

    # -------------------------------------------------------------------------
    # Versioned artifact subdirectories
    # -------------------------------------------------------------------------

    @property
    def embeddings_dir(self) -> Path:
        """Embeddings directory for the active model version."""
        return self._active_version_dir / "embeddings"

    @property
    def attention_dir(self) -> Path:
        """Attention components directory for the active model version."""
        return self._active_version_dir / "attention"

    @property
    def scoring_dir(self) -> Path:
        """Scoring models directory for the active model version."""
        return self._active_version_dir / "scoring"

    @property
    def data_dir(self) -> Path:
        """Data snapshot directory for the active model version."""
        return self._active_version_dir / "data"

    @property
    def staging_data_dir(self) -> Path:
        """
        Data snapshot directory inside staging.

        This is the write target for export_training_data.py and the read
        source for all training scripts during a training run. It is distinct
        from data_dir, which resolves through the active version and is used
        by the runtime loaders.
        """
        return self.staging_dir / "data"

    # -------------------------------------------------------------------------
    # Embeddings (vector representations)
    # -------------------------------------------------------------------------

    @property
    def book_subject_embeddings(self) -> Path:
        """Attention-pooled book embeddings derived from subjects."""
        return self.embeddings_dir / "book_subject_embeddings.npy"

    @property
    def book_subject_ids(self) -> Path:
        """Item indices corresponding to book_subject_embeddings."""
        return self.embeddings_dir / "book_subject_ids.json"

    @property
    def book_als_factors(self) -> Path:
        """ALS collaborative filtering latent factors for books."""
        return self.embeddings_dir / "book_als_factors.npy"

    @property
    def book_als_ids(self) -> Path:
        """Item indices corresponding to book_als_factors."""
        return self.embeddings_dir / "book_als_ids.json"

    @property
    def user_als_factors(self) -> Path:
        """ALS collaborative filtering latent factors for users."""
        return self.embeddings_dir / "user_als_factors.npy"

    @property
    def user_als_ids(self) -> Path:
        """User IDs corresponding to user_als_factors."""
        return self.embeddings_dir / "user_als_ids.json"

    # -------------------------------------------------------------------------
    # Attention (attention pooling components)
    # -------------------------------------------------------------------------

    def get_attention_path(self, strategy: AttentionStrategy) -> Path:
        """
        Get path for attention model components by strategy name.

        Args:
            strategy: One of 'scalar', 'perdim', 'selfattn', 'selfattn_perdim'.

        Returns:
            Path to the attention model file.

        Raises:
            ValueError: If strategy is not recognized.
        """
        valid_strategies = {"scalar", "perdim", "selfattn", "selfattn_perdim"}

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown attention strategy: '{strategy}'. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )

        return self.attention_dir / f"subject_attention_{strategy}.pth"

    @property
    def subject_attention_scalar(self) -> Path:
        """Scalar attention pooling components."""
        return self.attention_dir / "subject_attention_scalar.pth"

    @property
    def subject_attention_perdim(self) -> Path:
        """Per-dimension attention pooling components."""
        return self.attention_dir / "subject_attention_perdim.pth"

    @property
    def subject_attention_selfattn(self) -> Path:
        """Self-attention pooling components."""
        return self.attention_dir / "subject_attention_selfattn.pth"

    @property
    def subject_attention_selfattn_perdim(self) -> Path:
        """Self-attention with per-dimension pooling components."""
        return self.attention_dir / "subject_attention_selfattn_perdim.pth"

    # -------------------------------------------------------------------------
    # Scoring (ranking and scoring models)
    # -------------------------------------------------------------------------

    @property
    def bayesian_scores(self) -> Path:
        """Precomputed Bayesian popularity scores for books."""
        return self.scoring_dir / "bayesian_scores.npy"

    # -------------------------------------------------------------------------
    # Semantic indexes (FAISS-based, not versioned)
    # -------------------------------------------------------------------------

    @property
    def semantic_index_baseline(self) -> Path:
        """Baseline semantic search index directory."""
        return self.semantic_indexes_dir / "baseline"

    @property
    def semantic_index_baseline_clean(self) -> Path:
        """Baseline clean semantic search index directory."""
        return self.semantic_indexes_dir / "baseline_clean"

    @property
    def semantic_index_enriched_v1(self) -> Path:
        """Enriched v1 semantic search index directory."""
        return self.semantic_indexes_dir / "enriched_v1"

    @property
    def semantic_index_enriched_v1_subjects(self) -> Path:
        """Enriched v1 subjects semantic search index directory."""
        return self.semantic_indexes_dir / "enriched_v1_subjects"

    @property
    def semantic_index_enriched_v2(self) -> Path:
        """Enriched v2 semantic search index directory."""
        return self.semantic_indexes_dir / "enriched_v2"

    @property
    def semantic_index_enriched_v2_subjects(self) -> Path:
        """Enriched v2 subjects semantic search index directory."""
        return self.semantic_indexes_dir / "enriched_v2_subjects"

    # -------------------------------------------------------------------------
    # Data snapshots (versioned)
    # -------------------------------------------------------------------------

    @property
    def data_interactions(self) -> Path:
        """User-book interaction records for the active model version."""
        return self.data_dir / "interactions.pkl"

    @property
    def data_users(self) -> Path:
        """User metadata for the active model version."""
        return self.data_dir / "users.pkl"

    @property
    def data_books(self) -> Path:
        """Book metadata for the active model version."""
        return self.data_dir / "books.pkl"

    @property
    def data_book_subjects(self) -> Path:
        """Book-to-subject mappings for the active model version."""
        return self.data_dir / "book_subjects.pkl"

    @property
    def data_user_fav_subjects(self) -> Path:
        """User favourite subjects for the active model version."""
        return self.data_dir / "user_fav_subjects.pkl"

    @property
    def data_subjects(self) -> Path:
        """Subject vocabulary for the active model version."""
        return self.data_dir / "subjects.pkl"

    # -------------------------------------------------------------------------
    # Directory creation helpers
    # -------------------------------------------------------------------------

    def ensure_artifact_dirs(self) -> None:
        """
        Create top-level artifact infrastructure directories.

        Does not create versioned subdirectories — those are created by the
        artifact registry when a version is registered or promoted.
        """
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_indexes_dir.mkdir(parents=True, exist_ok=True)

    def ensure_staging_dirs(self) -> None:
        """Create all expected subdirectories inside staging."""
        for subdir in _VERSIONED_SUBDIRS:
            (self.staging_dir / subdir).mkdir(parents=True, exist_ok=True)


# Global singleton instance
PATHS = ModelPaths()
