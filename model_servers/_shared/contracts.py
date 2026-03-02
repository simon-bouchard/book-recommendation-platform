# model_servers/_shared/contracts.py
"""
Pydantic contracts for all model server HTTP APIs.

Single source of truth shared by server implementations and application clients.
Both sides import from here to guarantee request/response schema consistency.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# Shared primitives
# ===========================================================================


class HealthResponse(BaseModel):
    """Liveness/readiness check response returned by every model server."""

    status: str = Field(..., description="'ok' when server is ready to serve requests")
    server: str = Field(..., description="Server identifier, e.g. 'embedder'")
    artifact_version: str = Field(..., description="Version string of currently loaded artifacts")


class ScoredItem(BaseModel):
    """A single (item_idx, score) pair returned by similarity and recommendation operations."""

    item_idx: int
    score: float


class BookMeta(BaseModel):
    """
    Book metadata returned by enrich and popular operations.

    All optional fields may be absent for books with incomplete metadata.
    """

    item_idx: int
    title: str
    author: str | None = None
    year: int | None = None
    isbn: str | None = None
    cover_id: str | None = None
    avg_rating: float | None = None
    num_ratings: int = 0
    bayes_score: float | None = None


# ===========================================================================
# Embedder server
# ===========================================================================


class EmbedRequest(BaseModel):
    """
    Compute a normalized query embedding from a list of subject indices.

    Used by the application layer before calling subject_recs on the
    similarity server, or for any other embedding-dependent operation.
    """

    subject_indices: list[int] = Field(
        ...,
        min_length=1,
        description="Subject indices representing user preferences",
    )

    @field_validator("subject_indices")
    @classmethod
    def must_be_non_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("subject_indices must contain at least one index")
        return v


class EmbedResponse(BaseModel):
    """Normalized embedding vector as a flat list of floats."""

    vector: list[float] = Field(
        ..., description="L2-normalized embedding vector, length equals model embedding_dim"
    )


# ===========================================================================
# Similarity server
# ===========================================================================


class SubjectSimRequest(BaseModel):
    """HNSW nearest-neighbour lookup using the stored subject embedding for a book."""

    item_idx: int
    k: int = Field(default=200, ge=1, le=1000)


class AlsSimRequest(BaseModel):
    """HNSW nearest-neighbour lookup using the stored ALS factor for a book."""

    item_idx: int
    k: int = Field(default=200, ge=1, le=1000)


class HybridSimRequest(BaseModel):
    """
    Single-pass joint matmul over subject embeddings and ALS factors.

    Alpha controls the blend: 0.0 = pure subject, 1.0 = pure ALS.
    """

    item_idx: int
    k: int = Field(default=200, ge=1, le=1000)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


class SimResponse(BaseModel):
    """Ordered list of similar items with scores, highest score first."""

    results: list[ScoredItem]


class SubjectRecsRequest(BaseModel):
    """
    Joint scoring over all book subject embeddings and Bayesian popularity scores.

    Score = alpha * cosine(user_vector, book_embedding) + (1 - alpha) * bayesian_score

    Both scores are normalized before blending so neither dominates by scale.
    Alpha=1.0 is pure subject similarity, alpha=0.0 is pure popularity.
    """

    user_vector: list[float] = Field(
        ...,
        min_length=1,
        description="Normalized user embedding from the embedder server",
    )
    k: int = Field(default=200, ge=1, le=1000)
    alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Subject weight; popularity weight is (1 - alpha)",
    )


class SubjectRecsResponse(BaseModel):
    """Ordered list of recommended items with blended scores, highest score first."""

    results: list[ScoredItem]


# ===========================================================================
# ALS recommendation server
# ===========================================================================


class HasAlsUserRequest(BaseModel):
    """Check whether a user has ALS factors (warm/cold gate)."""

    user_id: int


class HasAlsUserResponse(BaseModel):
    """Result of the warm/cold user gate check."""

    user_id: int
    is_warm: bool


class HasBookAlsRequest(BaseModel):
    """Check whether a book has a normalized ALS factor in the similarity server."""

    item_idx: int


class HasBookAlsResponse(BaseModel):
    """Result of the book ALS membership check."""

    item_idx: int
    has_als: bool


class AlsRecsRequest(BaseModel):
    """Raw dot product recommendation: book_factors @ user_vector."""

    user_id: int
    k: int = Field(default=200, ge=1, le=1000)


class AlsRecsResponse(BaseModel):
    """
    Ordered list of recommended items.

    Scores are raw dot products — magnitudes carry meaning (predicted
    interaction strength) and must not be interpreted as probabilities.
    """

    results: list[ScoredItem]


# ===========================================================================
# Metadata server
# ===========================================================================


class EnrichRequest(BaseModel):
    """Fetch full book metadata for a list of item indices."""

    item_indices: list[int] = Field(..., min_length=1, max_length=2000)


class EnrichResponse(BaseModel):
    """
    Metadata for the requested items.

    Items missing from the metadata store are silently omitted.
    The response list may be shorter than the request list.
    """

    books: list[BookMeta]


class PopularRequest(BaseModel):
    """Retrieve the top-k books ranked by precomputed Bayesian score."""

    k: int = Field(default=100, ge=1, le=1000)


class PopularResponse(BaseModel):
    """Top-k books ordered by Bayesian score descending."""

    books: list[BookMeta]


class SubjectSearchRequest(BaseModel):
    """
    Resolve free-text subject phrases to candidate subject indices.

    Used by the chatbot to translate natural language subject mentions
    into subject_idx values suitable for embedding or filtering.
    """

    phrases: list[str] = Field(..., min_length=1, max_length=20)
    top_k_per_phrase: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate subject indices to return per phrase",
    )


class SubjectMatch(BaseModel):
    """A single phrase-to-subject resolution result."""

    phrase: str
    subject_idx: int
    subject_name: str
    score: float


class SubjectSearchResponse(BaseModel):
    """
    Candidate subject matches grouped by input phrase.

    Phrases with no match above the similarity threshold are omitted.
    """

    matches: list[SubjectMatch]
