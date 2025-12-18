# models/domain/config.py
"""
Configuration objects for recommendation requests with built-in validation.
"""

from dataclasses import dataclass, field
from typing import Literal

RecommendationMode = Literal["auto", "subject", "behavioral"]


@dataclass
class HybridConfig:
    """
    Configuration for hybrid candidate generation.

    Defines how to blend scores and how many candidates to generate
    from each source (hybrid, pure subject, pure popularity).

    Attributes:
        subject_weight: Weight for subject-based scores in hybrid blend (0-1)
        k_hybrid: Number of hybrid candidates (blended subject + popularity)
        k_subject: Number of pure subject-based candidates (optional, for diversity)
        k_popularity: Number of pure popularity-based candidates (optional, for diversity)

    Example:
        # Default: 200 hybrid candidates only
        config = HybridConfig(subject_weight=0.6)

        # With diversity from pure sources
        config = HybridConfig(
            subject_weight=0.6,
            k_hybrid=150,
            k_subject=25,      # Add some pure subject matches
            k_popularity=25    # Add some popular books
        )
    """

    subject_weight: float = 0.6
    k_hybrid: int = 200
    k_subject: int = 0
    k_popularity: int = 0

    @property
    def popularity_weight(self) -> float:
        """Popularity weight is automatically derived as 1 - subject_weight."""
        return 1.0 - self.subject_weight

    @property
    def total_candidates(self) -> int:
        """Total number of candidates that will be generated."""
        return self.k_hybrid + self.k_subject + self.k_popularity

    def __post_init__(self):
        """Validate hybrid configuration."""
        if not 0 <= self.subject_weight <= 1:
            raise ValueError(f"subject_weight must be in [0, 1], got {self.subject_weight}")

        if self.k_hybrid < 0:
            raise ValueError(f"k_hybrid must be non-negative, got {self.k_hybrid}")

        if self.k_subject < 0:
            raise ValueError(f"k_subject must be non-negative, got {self.k_subject}")

        if self.k_popularity < 0:
            raise ValueError(f"k_popularity must be non-negative, got {self.k_popularity}")

        if self.total_candidates == 0:
            raise ValueError("At least one of k_hybrid, k_subject, or k_popularity must be > 0")


@dataclass
class RecommendationConfig:
    """
    Configuration for a recommendation request.

    Defines what kind of recommendations to generate and how many.

    Attributes:
        k: Number of recommendations to return
        mode: Recommendation mode (auto/subject/behavioral)
        hybrid_config: Configuration for hybrid blending (if applicable)
    """

    k: int = 200
    mode: RecommendationMode = "auto"
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)

    def __post_init__(self):
        """Validate recommendation configuration."""
        if not 1 <= self.k <= 500:
            raise ValueError(f"k must be in [1, 500], got {self.k}")

        if self.mode not in ("auto", "subject", "behavioral"):
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'auto', 'subject', or 'behavioral'"
            )

        if not isinstance(self.hybrid_config, HybridConfig):
            raise TypeError(
                f"hybrid_config must be HybridConfig instance, got {type(self.hybrid_config)}"
            )

    @classmethod
    def default(cls, k: int = 200, mode: RecommendationMode = "auto") -> "RecommendationConfig":
        """
        Create a configuration with default settings.

        Args:
            k: Number of recommendations (default: 200)
            mode: Recommendation mode (default: "auto")

        Returns:
            RecommendationConfig with default hybrid settings
        """
        return cls(k=k, mode=mode, hybrid_config=HybridConfig())
