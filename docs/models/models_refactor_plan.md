# ML Models Module: Complete Refactor & Redesign Plan

## Executive Summary

**Goal**: Transform the models module from a poorly-structured, slow codebase into a clean, performant, testable system.

**Scope**: 
- Architectural redesign (6 layers with clear boundaries)
- Artifact renaming (descriptive, conventional names)
- Performance optimization (70% latency reduction for warm users)
- Testability (singleton + injectable pattern)
- File reorganization (logical structure)

**Timeline**: 4 weeks (big bang migration, no backward compatibility)

**Key Metrics**:
- Warm user latency: 70ms → 20ms (70% improvement)
- Memory usage: -200MB (remove unused GBT models)
- Test coverage: 0% → 80% (newly testable)
- Largest file: 600 lines → 300 lines (better organization)

---

## Current State Analysis

### Problems Identified

| Category | Problem | Impact |
|----------|---------|--------|
| **Architecture** | Strategy pattern that isn't polymorphic | Confusing code, fake abstraction |
| **Architecture** | Scattered business logic across 4 layers | Hard to understand flow |
| **Architecture** | NoOpReranker does filtering | Misleading name, hidden logic |
| **Performance** | Warm users compute unused embeddings | 50ms wasted per request |
| **Naming** | book_embs.npy is ambiguous | Don't know which embeddings |
| **Naming** | book_als_emb should be "factors" | Technically incorrect |
| **Naming** | bayesian_tensor should be "scores" | Not descriptive |
| **Structure** | ModelStore is god object (500+ lines) | Hard to maintain |
| **Structure** | shared_utils.py is god module | Everything in one file |
| **Testability** | Singletons can't be mocked | Can't unit test |
| **Coupling** | Generators take unused db: Session | Poor interface design |
| **Logic** | is_fallback flag passed everywhere | Wrong abstraction level |
| **Logic** | num_ratings threshold in API layer | Business logic in wrong place |

---

## Target Architecture

### 6-Layer Design

```
┌─────────────────────────────────────────────────────┐
│ Layer 6: API (HTTP Interface)                       │
│  - Routes, request/response formatting              │
│  - Authentication                                   │
└────────────────────┬────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────┐
│ Layer 5: Service (Business Rules)                   │
│  - RecommendationService                            │
│  - SimilarityService                                │
│  - Business logic: warm/cold, blending, fallback    │
└────────────────────┬────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────┐
│ Layer 4: Domain (Pipeline & Orchestration)          │
│  - RecommendationPipeline                           │
│  - CandidateGenerator (interface + implementations) │
│  - Filter, Ranker                                   │
└────────────────────┬────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────┐
│ Layer 3: Infrastructure (Computation)               │
│  - SubjectEmbedder (singleton + injectable)         │
│  - ALSModel (singleton + injectable)                │
│  - SimilarityIndex (FAISS wrapper)                  │
└────────────────────┬────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────┐
│ Layer 2: Repositories (Data Access)                 │
│  - EmbeddingRepository                              │
│  - MetadataRepository                               │
│  - ScoringRepository                                │
└────────────────────┬────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────┐
│ Layer 1: Domain Models (Business Objects)           │
│  - User, Candidate, RecommendedBook                 │
│  - Configuration objects                            │
└────────────────────┬────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────┐
│ Layer 0: Foundation (Constants, Paths, Config)      │
│  - PAD_IDX, paths, artifact locations               │
└─────────────────────────────────────────────────────┘
```

### Layer Contracts (Interfaces)

#### Layer 1: Domain Models
```python
# Contracts: Pure data classes, no dependencies

@dataclass
class User:
    user_id: int
    fav_subjects: list[int]
    # Properties: is_warm (checks ALS), has_preferences
    
@dataclass 
class Candidate:
    item_idx: int
    score: float
    source: str
    
@dataclass
class RecommendationConfig:
    k: int
    mode: str
    hybrid_config: HybridConfig
```

#### Layer 2: Repository Contracts
```python
# Contracts: Data loading, singleton for caching

class EmbeddingRepository:
    def get_book_subject_embeddings() -> tuple[np.ndarray, list[int]]
    def get_als_factors() -> tuple[user_factors, book_factors, user_ids, book_ids]
    
class MetadataRepository:
    def get_book_meta() -> pd.DataFrame
    def get_user_meta() -> pd.DataFrame
    
class ScoringRepository:
    def get_bayesian_scores() -> np.ndarray
```

#### Layer 3: Infrastructure Contracts
```python
# Contracts: Pure computation, singleton + injectable

class SubjectEmbedder:
    def __init__(strategy: str = "scalar", pooler = None)  # Injectable for tests
    def embed(subjects: list[int]) -> np.ndarray
    def embed_batch(subjects_list: list[list[int]]) -> np.ndarray
    
class ALSModel:
    def __init__(..., test_data = None)  # Injectable for tests
    def has_user(user_id: int) -> bool
    def has_book(item_idx: int) -> bool
    def recommend(user_id: int, k: int) -> list[int]
    
class SimilarityIndex:
    def __init__(embeddings: np.ndarray, ids: list[int], normalize: bool)
    def search(query: np.ndarray, k: int, exclude_id: int = None) -> tuple[scores, ids]
    def has_item(item_id: int) -> bool
```

#### Layer 4: Domain Contracts
```python
# Contracts: Business logic, no DB except filters

class CandidateGenerator(ABC):
    @abstractmethod
    def generate(user: User, k: int) -> list[Candidate]
    
    @property
    @abstractmethod
    def name() -> str

class Filter(Protocol):
    def apply(candidates: list[Candidate], user: User, db: Session) -> list[Candidate]
    
class Ranker(Protocol):
    def rank(candidates: list[Candidate], user: User) -> list[Candidate]

class RecommendationPipeline:
    def __init__(generator, fallback_generator, filter, ranker)
    def recommend(user: User, k: int, db: Session) -> list[Candidate]
```

#### Layer 5: Service Contracts
```python
# Contracts: Orchestration, business rules

class RecommendationService:
    def recommend(
        user: User,
        config: RecommendationConfig,
        db: Session
    ) -> list[RecommendedBook]
    
class SimilarityService:
    def get_similar(
        item_idx: int,
        mode: str,
        k: int,
        alpha: float = 0.6
    ) -> list[dict]
```

#### Layer 6: API Contracts
```python
# Contracts: HTTP interface

@router.post('/recommend')
def recommend(request: RecommendationRequest, db: Session) -> dict

@router.get('/book/{item_idx}/similar')
def get_similar(item_idx: int, mode: str, k: int) -> dict
```

---

## Artifact Renaming

### Complete Renaming Map

| Current Path | New Path | Reason |
|-------------|----------|--------|
| **Embeddings** | | |
| `models/data/book_embs.npy` | `models/artifacts/embeddings/book_subject_embeddings.npy` | Clarify: subject-pooled |
| `models/data/book_ids.json` | `models/artifacts/embeddings/book_subject_ids.json` | Match embeddings file |
| `models/data/book_als_emb.npy` | `models/artifacts/embeddings/book_als_factors.npy` | Technical accuracy |
| `models/data/book_als_ids.json` | `models/artifacts/embeddings/book_als_ids.json` | Move to new location |
| `models/data/user_als_emb.npy` | `models/artifacts/embeddings/user_als_factors.npy` | Technical accuracy |
| `models/data/user_als_ids.json` | `models/artifacts/embeddings/user_als_ids.json` | Move to new location |
| **Attention** | | |
| `models/data/subject_attention_components.pth` | `models/artifacts/attention/subject_attention_scalar.pth` | Remove redundant "components" |
| `models/data/subject_attention_components_perdim.pth` | `models/artifacts/attention/subject_attention_perdim.pth` | Remove redundant "components" |
| `models/data/subject_attention_components_selfattn.pth` | `models/artifacts/attention/subject_attention_selfattn.pth` | Remove redundant "components" |
| `models/data/subject_attention_components_selfattn_perdim.pth` | `models/artifacts/attention/subject_attention_selfattn_perdim.pth` | Remove redundant "components" |
| **Scoring** | | |
| `models/data/bayesian_tensor.npy` | `models/artifacts/scoring/bayesian_scores.npy` | More descriptive |
| `models/data/gbt_cold.pickle` | `models/artifacts/scoring/gbt_cold.pickle` | Organize by type |
| `models/data/gbt_warm.pickle` | `models/artifacts/scoring/gbt_warm.pickle` | Organize by type |

### Naming Convention

**Pattern**: `{entity}_{representation_type}_{optional_variant}.{ext}`

**Examples**:
- `book_subject_embeddings.npy` - entity=book, type=subject embeddings
- `book_als_factors.npy` - entity=book, type=ALS factors
- `user_als_factors.npy` - entity=user, type=ALS factors
- `bayesian_scores.npy` - purpose-based (scores)
- `subject_attention_scalar.pth` - subject attention, scalar variant

---

## Python Variable Naming (Consistency)

**Critical**: Variable names must match artifact names for clarity!

### Quick Reference: Variable Renamings

| Old Variable Name | New Variable Name | Reason |
|------------------|-------------------|--------|
| `book_embs` | `book_subject_embeddings` | Clarify which embeddings |
| `book_ids` | `book_subject_ids` | Match with embedding type |
| `book_als_emb` | `book_als_factors` | Technical accuracy (ALS = factorization) |
| `user_als_embs` | `user_als_factors` | Technical accuracy |
| `bayesian_tensor` | `bayesian_scores` | More descriptive (it's scores, not tensor) |
| `load_book_embeddings()` | `get_book_subject_embeddings()` | Explicit about type |
| `get_als_embeddings()` | `get_als_factors()` | Technical accuracy |
| `get_bayesian_tensor()` | `get_bayesian_scores()` | More descriptive |
| `subject_emb` | `pooler` | More accurate (it's an attention pooler) |

### Complete Variable Renaming Map

#### Current State (Confusing)
```python
# In shared_utils.py / repositories
self._book_embs           # Which embeddings? Subject? ALS? Semantic?
self._book_ids            # Which books? From which model?
self._book_als_emb        # Should be "factors" not "emb"
self._user_als_embs       # Should be "factors" not "embs"
book_embs, book_ids       # Return values are unclear

# In functions
def load_book_embeddings() -> tuple[np.ndarray, list]:
    # What kind of embeddings?
```

#### Target State (Clear)
```python
# In EmbeddingRepository
self._book_subject_embeddings      # ✅ Clear: subject-pooled embeddings
self._book_subject_ids             # ✅ Clear: IDs for subject embeddings
self._book_als_factors             # ✅ Clear: ALS factorization (not embeddings)
self._book_als_ids                 # ✅ Clear: IDs for ALS factors
self._user_als_factors             # ✅ Clear: user factors from ALS
self._user_als_ids                 # ✅ Clear: IDs for user factors

# Return values match
def get_book_subject_embeddings() -> tuple[np.ndarray, list[int]]:
    return self._book_subject_embeddings, self._book_subject_ids

def get_als_factors() -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    return (
        self._user_als_factors,    # Clear what each return value is
        self._book_als_factors,
        self._user_als_ids,
        self._book_als_ids
    )
```

### Repository Method Names

#### EmbeddingRepository
```python
# OLD (from shared_utils.py)
load_book_embeddings()              # Which embeddings?
get_als_embeddings()                # Returns factors, not embeddings!

# NEW (clear and consistent)
get_book_subject_embeddings()       # Returns subject-pooled embeddings
get_als_factors()                   # Returns ALS factorization matrices
```

#### MetadataRepository  
```python
# OLD
get_book_meta()                     # OK, clear enough
get_user_meta()                     # OK, clear enough

# NEW (no change needed)
get_book_meta()
get_user_meta()
```

#### ScoringRepository
```python
# OLD
get_bayesian_tensor()               # "Tensor" is misleading (it's a numpy array)

# NEW
get_bayesian_scores()               # ✅ Clear: these are Bayesian popularity scores
```

### Infrastructure Class Attributes

#### SubjectEmbedder
```python
# OLD
self.subject_emb                    # OK, but singular

# NEW  
self.pooler                         # ✅ More accurate (it's an attention pooler)
```

#### ALSModel
```python
# OLD (from shared_utils.py)
self._user_als_embs                 # Should be "factors"
self._book_als_embs                 # Should be "factors"

# NEW
self.user_factors                   # ✅ Technically accurate
self.book_factors                   # ✅ Technically accurate
self.user_id_to_row                 # ✅ Clear mapping
self.book_row_to_id                 # ✅ Clear mapping
```

#### SimilarityIndex
```python
# NEW (no old equivalent)
self.ids                            # Item IDs in the index
self.id_to_row                      # Mapping from item_idx to row
self.index                          # FAISS index
```

### Domain Model Attributes

#### User
```python
# OLD (from ORM + scattered logic)
user.fav_subjects_idxs              # Inconsistent naming

# NEW
user.fav_subjects                   # ✅ Consistent (already plural)
```

#### Candidate
```python
# NEW (no old equivalent)
candidate.item_idx                  # Book ID
candidate.score                     # Relevance score
candidate.source                    # Which generator produced it
```

### Local Variable Conventions

```python
# In generators and services

# OLD patterns (inconsistent)
embs = ...
book_embs = ...
als_emb = ...
embedding = ...

# NEW patterns (explicit)
subject_embeddings = ...            # When working with subject embeddings
als_factors = ...                   # When working with ALS factors  
user_embedding = ...                # When computing for a single user
book_embedding = ...                # When looking up for a single book
```

### Function Parameter Names

```python
# OLD (from various files)
def generate(self, user_emb: np.ndarray):
    # Is this subject embedding or ALS?

def get_similar(item_idx: int, mode: str):
    # OK

# NEW (explicit when needed)
def embed(self, subjects: list[int]) -> np.ndarray:
    # Returns subject embedding (clear from context)

def recommend(self, user_id: int, k: int) -> list[int]:
    # ALS recommendation (clear from ALSModel context)
```

### Naming Rules

1. **File names match variable names**
   - `book_subject_embeddings.npy` ↔ `self._book_subject_embeddings`
   
2. **Method names match return types**
   - `get_book_subject_embeddings()` returns `(subject_embeddings, subject_ids)`
   
3. **Use technically accurate terms**
   - ALS produces "factors" not "embeddings"
   - Bayesian produces "scores" not "tensor"
   
4. **Be explicit about entity and type**
   - `book_subject_embeddings` not just `book_embs`
   - `user_als_factors` not just `user_embs`
   
5. **Local variables can be shorter if context is clear**
   - Inside `SubjectEmbedder.embed()`: `embedding` is fine (context clear)
   - Inside `ALSModel.recommend()`: `factors` is fine (context clear)
   - At service layer: `subject_embeddings` is better (multiple types present)

### Migration Checklist for Variables

- [ ] Rename all `*_emb` ALS variables to `*_factors`
- [ ] Rename `book_embs` to `book_subject_embeddings`
- [ ] Rename `book_ids` to `book_subject_ids` (when referring to subject embeddings)
- [ ] Rename `bayesian_tensor` to `bayesian_scores`
- [ ] Update all function signatures to match new names
- [ ] Update all docstrings to reflect new terminology
- [ ] Search codebase for old variable patterns and update

### Impact on Code Readability

**Before** (ambiguous):
```python
book_embs, book_ids = load_book_embeddings()
als_embs = get_als_embeddings()
user_emb = compute_user_emb(subjects)
```
🤔 What kind of embeddings? Can't tell without reading docs.

**After** (self-documenting):
```python
subject_embeddings, subject_ids = repo.get_book_subject_embeddings()
user_factors, book_factors, user_ids, book_ids = repo.get_als_factors()
user_embedding = embedder.embed(subjects)
```
✅ Immediately clear what each variable contains!

---

## File Organization

### Current Structure (Before)
```
models/
├── data/                           # 13 files mixed together
│   ├── book_embs.npy
│   ├── book_ids.json
│   ├── book_als_emb.npy
│   ├── bayesian_tensor.npy
│   ├── gbt_cold.pickle
│   └── ... (8 more files)
├── shared_utils.py                 # 500+ lines god module
├── book_similarity_engine.py       # Singleton strategies
├── candidate_generators.py
├── rerankers.py
├── recommendation_engine.py
├── recommender_strategy.py
└── engines_reload.py
```

### Target Structure (After)
```
models/
├── artifacts/                      # NEW: Organized by type
│   ├── embeddings/
│   │   ├── book_subject_embeddings.npy
│   │   ├── book_subject_ids.json
│   │   ├── book_als_factors.npy
│   │   ├── book_als_ids.json
│   │   ├── user_als_factors.npy
│   │   └── user_als_ids.json
│   ├── attention/
│   │   ├── subject_attention_scalar.pth
│   │   ├── subject_attention_perdim.pth
│   │   ├── subject_attention_selfattn.pth
│   │   └── subject_attention_selfattn_perdim.pth
│   └── scoring/
│       ├── bayesian_scores.npy
│       ├── gbt_cold.pickle
│       └── gbt_warm.pickle
│
├── core/                           # NEW: Foundation
│   ├── __init__.py
│   ├── constants.py                # PAD_IDX
│   ├── paths.py                    # All artifact paths
│   └── config.py                   # Environment config
│
├── domain/                         # NEW: Business objects
│   ├── __init__.py
│   ├── user.py                     # User domain model
│   ├── recommendation.py           # Candidate, RecommendedBook
│   ├── config.py                   # Configuration objects
│   ├── candidate_generation.py    # CandidateGenerator + implementations
│   ├── filters.py                  # ReadBooksFilter, etc
│   ├── rankers.py                  # NoOpRanker, etc
│   └── pipeline.py                 # RecommendationPipeline
│
├── repositories/                   # NEW: Data access
│   ├── __init__.py
│   ├── embedding_repository.py    # Load embeddings
│   ├── metadata_repository.py     # Load book/user metadata
│   └── scoring_repository.py      # Load Bayesian scores
│
├── infrastructure/                 # NEW: Computation primitives
│   ├── __init__.py
│   ├── subject_embedder.py        # SubjectEmbedder (singleton + injectable)
│   ├── als_model.py               # ALSModel (singleton + injectable)
│   └── similarity_index.py        # SimilarityIndex (FAISS wrapper)
│
├── services/                       # NEW: Business rules
│   ├── __init__.py
│   ├── recommendation_service.py  # Main recommendation service
│   └── similarity_service.py      # Book similarity service
│
├── subject_attention_strategy.py  # KEEP: Attention implementations
│
└── training/                       # UPDATED: Training scripts
    ├── data/                       # Training input (unchanged location)
    ├── export_training_data.py    # UPDATED: Use new paths
    ├── precompute_embs.py         # UPDATED: Save to new paths
    ├── precompute_bayesian.py     # UPDATED: Save to new paths
    ├── train_als.py               # UPDATED: Save to new paths
    ├── train_subject_attention.py # UPDATED: Save to new paths
    ├── train_subjects_embs.py     # UPDATED: Save to new paths
    ├── train_subject_embs_contrastive.py  # UPDATED
    ├── train_cold_gbt.py          # UPDATED: Load from new paths
    └── train_warm_gbt.py          # UPDATED: Load from new paths
```

### Files Deleted
- `shared_utils.py` - Split into domain, repositories, infrastructure
- `recommendation_engine.py` - Replaced by pipeline.py + service
- `recommender_strategy.py` - Replaced by service layer
- `engines_reload.py` - Simplified (just repos.reset())
- `book_similarity_engine.py` - Moved to similarity_service.py
- `candidate_generators.py` - Moved to domain/candidate_generation.py
- `rerankers.py` - Replaced by filters.py + rankers.py

---

## Interface Schemas

### Key Data Structures

```python
# Layer 1: Domain Models

User {
    user_id: int
    fav_subjects: list[int]
    country?: str
    age?: int
    filled_age?: str
    
    # Computed properties
    is_warm: bool                    # Checks ALSModel().has_user()
    has_preferences: bool            # Checks fav_subjects validity
}

Candidate {
    item_idx: int
    score: float
    source: str                      # e.g., "als", "subject_similarity"
}

RecommendedBook {
    item_idx: int
    title: str
    author?: str
    year?: int
    isbn?: str
    cover_id?: str
    score: float
    avg_rating?: float
    num_ratings: int
}

RecommendationConfig {
    k: int
    mode: str                        # "auto" | "subject" | "behavioral"
    hybrid_config: HybridConfig
    
    def __post_init__():
        if not 1 <= k <= 500:
            raise ValueError("k must be in [1, 500]")
        if mode not in ("auto", "subject", "behavioral"):
            raise ValueError(f"Invalid mode: {mode}")
}

HybridConfig {
    subject_weight: float
    popularity_weight: float
    k_per_source: int
    dedup_strategy: str              # "highest_score" | "blend"
    
    def __post_init__():
        if not 0 <= subject_weight <= 1:
            raise ValueError("subject_weight must be in [0, 1]")
        if not 0 <= popularity_weight <= 1:
            raise ValueError("popularity_weight must be in [0, 1]")
        if abs(subject_weight + popularity_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to ~1.0")
        if dedup_strategy not in ("highest_score", "blend"):
            raise ValueError(f"Invalid dedup_strategy: {dedup_strategy}")
}
```

### Repository Interfaces

```python
# Layer 2: Data Access

EmbeddingRepository (singleton) {
    get_book_subject_embeddings() -> (embeddings: np.ndarray, ids: list[int])
    get_als_factors() -> (
        user_factors: np.ndarray,
        book_factors: np.ndarray,
        user_ids: list[int],
        book_ids: list[int]
    )
}

MetadataRepository (singleton) {
    get_book_meta() -> pd.DataFrame[item_idx]  # indexed by item_idx
    get_user_meta() -> pd.DataFrame[user_id]   # indexed by user_id
}

ScoringRepository (singleton) {
    get_bayesian_scores() -> np.ndarray  # aligned with book_subject_ids
}
```

### Infrastructure Interfaces

```python
# Layer 3: Computation

SubjectEmbedder (singleton + injectable) {
    __init__(strategy: str = "scalar", pooler = None)
    embed(subjects: list[int]) -> np.ndarray
    embed_batch(subjects_list: list[list[int]]) -> np.ndarray
    @classmethod reset()
}

ALSModel (singleton + injectable) {
    __init__(user_factors=None, book_factors=None, user_ids=None, book_ids=None)
    has_user(user_id: int) -> bool
    has_book(item_idx: int) -> bool
    recommend(user_id: int, k: int) -> list[int]
    @classmethod reset()
}

SimilarityIndex {
    __init__(embeddings: np.ndarray, ids: list[int], normalize: bool = True)
    search(query: np.ndarray, k: int, exclude_id: int = None) 
        -> (scores: np.ndarray, item_ids: np.ndarray)
    has_item(item_id: int) -> bool
}
```

### Domain Interfaces

```python
# Layer 4: Business Logic

CandidateGenerator (ABC) {
    @abstractmethod
    generate(user: User, k: int) -> list[Candidate]
    
    @property
    @abstractmethod
    name() -> str
}

# Implementations:
SubjectBasedGenerator(CandidateGenerator) {
    __init__(embedder: SubjectEmbedder, similarity: SimilarityIndex, config?)
    generate(user: User, k: int) -> list[Candidate]
    # Returns [] if user.has_preferences == False
}

ALSBasedGenerator(CandidateGenerator) {
    __init__(als_model: ALSModel)
    generate(user: User, k: int) -> list[Candidate]
    # Returns [] if not als.has_user(user.user_id)
}

BayesianPopularityGenerator(CandidateGenerator) {
    __init__(scores: np.ndarray, book_ids: list[int])
    generate(user: User, k: int) -> list[Candidate]
    # Always succeeds (fallback)
}

HybridGenerator(CandidateGenerator) {
    __init__(generators: list[tuple[CandidateGenerator, float]], config?)
    generate(user: User, k: int) -> list[Candidate]
    # Blends multiple sources
}

Filter (Protocol) {
    apply(candidates: list[Candidate], user: User, db: Session) -> list[Candidate]
}

Ranker (Protocol) {
    rank(candidates: list[Candidate], user: User) -> list[Candidate]
}

RecommendationPipeline {
    __init__(
        generator: CandidateGenerator,
        fallback_generator: CandidateGenerator,
        filter: Filter,
        ranker: Ranker
    )
    
    recommend(user: User, k: int, db: Session) -> list[Candidate]
    # Flow: generate -> fallback if needed -> filter -> rank -> top k
}
```

### Service Interfaces

```python
# Layer 5: Orchestration

RecommendationService {
    __init__()  # Lazy loads all infrastructure
    
    recommend(
        user: User,
        config: RecommendationConfig,
        db: Session
    ) -> list[RecommendedBook]
    
    # Business rules:
    # - if user.is_warm: use ALS
    # - elif user.has_preferences: use hybrid
    # - else: use popularity
    
    # Logging:
    # - Start: user_id, mode, is_warm, has_preferences
    # - Complete: user_id, mode, count, latency_ms
    # - Error: user_id, mode, error, traceback
}

# Logging Examples
"""
# Success case
logger.info("recommendation_started", 
    user_id=123, mode="auto", is_warm=True)
    
logger.info("recommendation_completed",
    user_id=123, mode="auto", count=50, latency_ms=23)

# Error case  
logger.error("recommendation_failed",
    user_id=123, mode="auto", error=str(e), 
    traceback=traceback.format_exc())
"""

SimilarityService {
    __init__()  # Lazy loads similarity indices
    
    get_similar(
        item_idx: int,
        mode: str,              # "subject" | "als" | "hybrid"
        k: int,
        alpha: float = 0.6
    ) -> list[dict]
    
    # Logging:
    # - Start: item_idx, mode
    # - Complete: item_idx, mode, count, latency_ms
}
```

### API Interfaces

```python
# Layer 6: HTTP

# Request Models (Pydantic)
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    k: int = Field(200, ge=1, le=500, description="Number of recommendations")
    mode: str = Field("auto", regex="^(auto|subject|behavioral)$")
    subject_weight: float = Field(0.6, ge=0, le=1)

# Response Models (Pydantic)
class RecommendedBookSchema(BaseModel):
    item_idx: int
    title: str
    author: Optional[str]
    year: Optional[int]
    isbn: Optional[str]
    cover_id: Optional[str]
    score: float
    avg_rating: Optional[float]
    num_ratings: int

class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedBookSchema]
    count: int
    config: dict
    user_type: str          # "warm" | "cold"
    user_has_preferences: bool

class SimilarBookSchema(BaseModel):
    item_idx: int
    title: str
    author: Optional[str]
    year: Optional[int]
    cover_id: Optional[str]
    score: float

class SimilarityResponse(BaseModel):
    similar_books: list[SimilarBookSchema]
    mode: str
    count: int

# Endpoints
POST /recommend {
    request: RecommendationRequest
    response: RecommendationResponse
}

GET /book/{item_idx}/similar {
    params: {
        item_idx: int (path)
        mode: str (query)
        k: int (query)
        alpha: float (query)
    }
    response: SimilarityResponse
}
```

---

## Migration Schedule

### Week 1: Foundation + Infrastructure

**Goal**: Build layers 0-3 with full test coverage

#### Day 1-2: Foundation & Repositories
- [ ] Create directory structure (`core/`, `repositories/`, `infrastructure/`, `domain/`, `services/`)
- [ ] Create `core/constants.py` (PAD_IDX)
- [ ] Create `core/paths.py` (all artifact paths with new names)
- [ ] Run artifact migration script (rename & move files)
- [ ] Create `repositories/embedding_repository.py`
  - Use new variable names: `_book_subject_embeddings`, `_book_als_factors`, `_user_als_factors`
  - Method names: `get_book_subject_embeddings()`, `get_als_factors()`
- [ ] Create `repositories/metadata_repository.py`
  - Use existing names: `_book_meta`, `_user_meta`
- [ ] Create `repositories/scoring_repository.py`
  - Use new variable name: `_bayesian_scores` (not `_bayesian_tensor`)
- [ ] Add validation to all configuration objects (`__post_init__`)
- [ ] Add Pydantic response models to `domain/recommendation.py`
- [ ] **Tests**: Repository loading, path validation, config validation

#### Day 3-4: Infrastructure
- [ ] Create `infrastructure/subject_embedder.py` (singleton + injectable)
- [ ] Create `infrastructure/als_model.py` (singleton + injectable)
- [ ] Create `infrastructure/similarity_index.py` (FAISS wrapper)
- [ ] **Tests**: Mock injections, singleton caching, computation correctness

#### Day 5: Integration Testing
- [ ] Test that repositories load from new paths
- [ ] Test that infrastructure uses repositories correctly
- [ ] Performance benchmarks (FAISS build time, embedding compute time)

**Deliverables**:
- ✅ Artifacts renamed and organized
- ✅ Foundation + repositories + infrastructure layers complete
- ✅ 80%+ test coverage on layers 0-3
- ✅ Old code still works (hasn't been touched yet)

---

### Week 2: Domain Logic

**Goal**: Build layer 4 with business logic

#### Day 1-2: Domain Models & Configuration
- [ ] Create `domain/user.py` (User domain model with is_warm, has_preferences)
- [ ] Create `domain/recommendation.py` (Candidate, RecommendedBook)
- [ ] Create `domain/config.py` (RecommendationConfig, HybridConfig)
- [ ] **Tests**: User properties (is_warm checks ALS), config validation

#### Day 3-4: Generators
- [ ] Create `domain/candidate_generation.py`
- [ ] Implement `CandidateGenerator` interface
- [ ] Implement `SubjectBasedGenerator`
- [ ] Implement `ALSBasedGenerator`
- [ ] Implement `BayesianPopularityGenerator`
- [ ] Implement `HybridGenerator`
- [ ] **Tests**: Each generator with mocked infrastructure

#### Day 5: Pipeline
- [ ] Create `domain/filters.py` (ReadBooksFilter)
- [ ] Create `domain/rankers.py` (NoOpRanker)
- [ ] Create `domain/pipeline.py` (RecommendationPipeline)
- [ ] **Tests**: Pipeline flow (generate → fallback → filter → rank)

**Deliverables**:
- ✅ Domain layer complete
- ✅ All generators testable with mocks
- ✅ Pipeline tested end-to-end with mock components
- ✅ No database coupling in generators

---

### Week 3: Service + API

**Goal**: Build layers 5-6 and switch over

#### Day 1-2: Services
- [ ] Create `services/recommendation_service.py`
  - Implement business rules (warm/cold decision, blending)
  - Add structured logging for all requests
    - Log start: user_id, mode, is_warm
    - Log completion: user_id, mode, count, latency_ms
    - Log errors with context
- [ ] Create `services/similarity_service.py`
  - Add logging for similarity requests
- [ ] **Tests**: Service with mocked domain layer, verify logging output

#### Day 3-4: API Layer
- [ ] Create Pydantic request models (`RecommendationRequest`)
- [ ] Create Pydantic response models (`RecommendationResponse`, `SimilarityResponse`)
- [ ] Create `routes/recommendations.py` (new design with Pydantic)
- [ ] Create `routes/similarity.py` (new design with Pydantic)
- [ ] Update `routes/api.py` to include new routers
- [ ] Keep old endpoints for comparison (side-by-side testing)
- [ ] **Tests**: API integration tests with test database, validate Pydantic schemas

#### Day 5: Validation & Performance
- [ ] Run old and new code side-by-side
- [ ] Compare outputs (should be identical)
- [ ] Measure latency (expect 70% improvement for warm users)
- [ ] Validate memory usage (expect -200MB)

**Deliverables**:
- ✅ Service layer complete
- ✅ New API endpoints working
- ✅ Performance validated
- ✅ Ready to switch over

---

### Week 4: Training Scripts + Cutover

**Goal**: Update training pipeline and remove old code

#### Day 1-2: Training Script Updates
- [ ] Update `export_training_data.py` (no changes needed)
- [ ] Update `precompute_embs.py` (save to new paths)
- [ ] Update `precompute_bayesian.py` (save to new paths)
- [ ] Update `train_als.py` (save to new paths)
- [ ] Update `train_subject_attention.py` (save to new paths)
- [ ] Update `train_subjects_embs.py` (save to new paths)
- [ ] Update `train_subject_embs_contrastive.py` (save to new paths)
- [ ] Update `train_cold_gbt.py` (load from new paths)
- [ ] Update `train_warm_gbt.py` (load from new paths)
- [ ] **Test**: Run each training script locally

#### Day 3: Automated Training Pipeline
- [ ] Update `ops/automated_training.py`:
  - Update artifact sync paths (models/artifacts instead of models/data)
  - Update file list to sync
  - Update reload endpoint call
- [ ] **Test**: Full training pipeline on Azure VM

#### Day 4: Cutover
- [ ] Remove old endpoints from `routes/api.py`
- [ ] Delete old files:
  - `shared_utils.py`
  - `recommendation_engine.py`
  - `recommender_strategy.py`
  - `engines_reload.py`
  - `book_similarity_engine.py`
  - `candidate_generators.py`
  - `rerankers.py`
- [ ] Update all imports in remaining files
- [ ] Create `models/engines_reload.py` (simplified version)

#### Day 5: Validation
- [ ] Full regression test suite
- [ ] Load testing (ensure performance improvements hold)
- [ ] Deploy to production
- [ ] Monitor for issues

**Deliverables**:
- ✅ Training scripts updated
- ✅ Automated training works end-to-end
- ✅ Old code removed
- ✅ System running in production

---

## Success Metrics

### Performance
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Warm user latency (p50) | 70ms | 20ms | 70% improvement ✅ |
| Warm user latency (p95) | 120ms | 40ms | 67% improvement ✅ |
| Cold user latency (p50) | 80ms | 80ms | No regression ✅ |
| Memory usage | 1.2GB | 1.0GB | -200MB ✅ |
| First request (cold start) | 2s | 2s | No regression ✅ |

### Code Quality
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Largest file | 600 lines | 300 lines | <300 lines ✅ |
| Files with >400 lines | 3 | 0 | 0 ✅ |
| Test coverage | 0% | 80% | >75% ✅ |
| Circular imports | 2 | 0 | 0 ✅ |
| God objects | 2 | 0 | 0 ✅ |

### Maintainability
| Metric | Before | After |
|--------|--------|-------|
| Can add new generator? | Hard (4 files) | Easy (1 file) |
| Can test generators? | No (singletons) | Yes (injectable) |
| Can understand flow? | Hard (scattered) | Easy (1 service) |
| Can add new feature? | 3-4 days | 1 day |

---

## Rollback Plan

### If Issues Found in Week 1-3
- Revert git branch (old code untouched)
- Continue development
- No impact on production

### If Issues Found After Week 4 Cutover
1. **Immediate**: Revert git commit (5 minutes)
2. **Short-term**: 
   - Keep old artifacts in `models/data_backup/`
   - Create symlinks if needed
3. **Investigation**: 
   - Check logs for errors
   - Compare outputs old vs new
   - Fix issues in development branch

### Backup Strategy
- Git tag before each week: `refactor-week-1`, `refactor-week-2`, etc.
- Keep old artifacts in `models/data_backup/` for 2 weeks after cutover
- Keep old code in git history (never force push)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression | Low | High | Benchmark each layer, compare with old code |
| Breaking training pipeline | Medium | High | Test each script locally before VM |
| Missing edge case | Medium | Medium | Comprehensive test suite, side-by-side comparison |
| Memory leak | Low | High | Monitor memory usage in production |
| Database query regression | Low | Medium | Reuse exact same queries in filter layer |

---

## Dependencies

### External Dependencies (unchanged)
- Python 3.10+
- PyTorch (for attention models)
- FAISS (for similarity search)
- NumPy, Pandas (for data manipulation)
- FastAPI (for API)
- SQLAlchemy (for database)

### Internal Dependencies
- `app/database.py` - Database connection (used by filters only)
- `app/table_models.py` - ORM models (only for DB queries)
- Training data in `models/training/data/` (read-only)
- Attention strategy implementations in `subject_attention_strategy.py`

---

## Decisions Made

### Q1: Should we add request validation to configuration objects?
**Decision**: ✅ Yes, add `__post_init__` validation
```python
@dataclass
class HybridConfig:
    subject_weight: float = 0.6
    popularity_weight: float = 0.4
    
    def __post_init__(self):
        if not 0 <= self.subject_weight <= 1:
            raise ValueError("subject_weight must be in [0, 1]")
        if not 0 <= self.popularity_weight <= 1:
            raise ValueError("popularity_weight must be in [0, 1]")
        if abs(self.subject_weight + self.popularity_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to ~1.0")
```

### Q2: Should API responses be Pydantic models or raw dicts?
**Decision**: ✅ Yes, Pydantic models for type safety
```python
class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedBook]
    count: int
    config: dict
    user_type: str
    user_has_preferences: bool
```

### Q3: Should we add logging/metrics at service layer?
**Decision**: ✅ Yes, add structured logging
```python
import time
from app.agents.logging import get_logger

logger = get_logger(__name__)

def recommend(self, user, config, db):
    start = time.time()
    logger.info("recommendation_started", 
        user_id=user.user_id, 
        mode=config.mode, 
        is_warm=user.is_warm
    )
    
    results = # ... do work
    
    logger.info("recommendation_completed",
        user_id=user.user_id,
        mode=config.mode,
        count=len(results),
        latency_ms=int((time.time() - start) * 1000)
    )
    return results
```

### Q4: Should we support multiple attention strategies simultaneously?
**Decision**: ✅ No, only one strategy at a time
- Keep it simple: one strategy configured via env var
- Can add strategy ensembles later if needed

### Q5: Should we add caching for repeated recommendations?
**Decision**: ❌ Not in this refactor, add as separate feature later
- Keep refactor focused on architecture
- Add Redis caching layer later if needed

---

## Appendix: Testing Strategy

### Unit Tests (Layer by Layer)

**Layer 0-1 (Foundation + Models)**
```python
def test_user_is_warm_checks_als():
    # Mock ALS to return True
    user = User(user_id=123, fav_subjects=[1,2,3])
    assert user.is_warm == True  # Because mock ALS has user

def test_user_has_preferences_with_pad():
    user = User(user_id=123, fav_subjects=[PAD_IDX])
    assert user.has_preferences == False
```

**Layer 2 (Repositories)**
```python
def test_embedding_repository_loads_from_new_path():
    repo = EmbeddingRepository()
    embs, ids = repo.get_book_subject_embeddings()
    assert embs.shape[0] == len(ids)
    assert PATHS.book_subject_embeddings.exists()
```

**Layer 3 (Infrastructure)**
```python
def test_subject_embedder_with_mock_pooler():
    mock_pooler = Mock(return_value=torch.tensor([[0.1, 0.2]]))
    embedder = SubjectEmbedder(pooler=mock_pooler)
    result = embedder.embed([1, 2, 3])
    assert result.shape == (2,)

def test_als_model_with_mock_factors():
    als = ALSModel(
        user_factors=np.random.randn(10, 64),
        book_factors=np.random.randn(100, 64),
        user_ids=list(range(10)),
        book_ids=list(range(100))
    )
    assert als.has_user(5) == True
    assert als.has_user(99) == False
```

**Layer 4 (Domain)**
```python
def test_subject_generator_returns_empty_without_preferences():
    mock_embedder = Mock()
    mock_similarity = Mock()
    generator = SubjectBasedGenerator(mock_embedder, mock_similarity)
    
    user = User(user_id=1, fav_subjects=[PAD_IDX])
    candidates = generator.generate(user, k=10)
    
    assert candidates == []
    mock_embedder.embed.assert_not_called()  # Didn't waste computation

def test_pipeline_uses_fallback_when_primary_fails():
    primary_gen = Mock(return_value=[])  # Returns empty
    fallback_gen = Mock(return_value=[Candidate(1, 0.9, "fallback")])
    
    pipeline = RecommendationPipeline(primary_gen, fallback_gen, NoFilter(), NoOpRanker())
    
    user = User(user_id=1, fav_subjects=[1,2,3])
    results = pipeline.recommend(user, k=10, db=mock_db)
    
    assert len(results) > 0
    fallback_gen.assert_called_once()
```

**Layer 5 (Service)**
```python
def test_service_uses_als_for_warm_user(mock_als_model):
    service = RecommendationService()
    # Inject mocks
    service._als_model = mock_als_model
    
    user = User(user_id=123, fav_subjects=[1,2,3])
    # user.is_warm will be True because mock_als has user
    
    config = RecommendationConfig.default()
    results = service.recommend(user, config, mock_db)
    
    mock_als_model.recommend.assert_called_once()
```

**Layer 6 (API)**
```python
def test_recommend_endpoint_returns_200(test_client):
    response = test_client.post("/recommend", json={
        "user_id": 123,
        "k": 10,
        "mode": "auto"
    })
    assert response.status_code == 200
    assert "recommendations" in response.json()
```

### Integration Tests

```python
def test_full_recommendation_flow():
    """End-to-end test with real repositories but test database"""
    # Use real embeddings, real ALS, test database
    user_orm = create_test_user_in_db()
    user = User.from_orm(user_orm)
    
    service = RecommendationService()
    config = RecommendationConfig.default(k=10)
    
    results = service.recommend(user, config, test_db)
    
    assert len(results) > 0
    assert all(isinstance(r, RecommendedBook) for r in results)
    # Results should be filtered (no books user already read)
```

---

## Summary

This plan combines:
1. ✅ **Architectural redesign** - 6 clear layers with defined contracts
2. ✅ **Artifact renaming** - Descriptive, conventional names (files AND variables)
3. ✅ **Variable naming consistency** - Python variables match artifact names
4. ✅ **File reorganization** - Logical structure
5. ✅ **Performance optimization** - 70% latency reduction for warm users
6. ✅ **Testability** - Singleton + injectable pattern
7. ✅ **Simplification** - Remove god objects, clarify responsibilities
8. ✅ **Training pipeline** - Updated to use new paths
9. ✅ **Configuration validation** - `__post_init__` checks
10. ✅ **Type-safe responses** - Pydantic models
11. ✅ **Observability** - Structured logging at service layer
12. ✅ **Single strategy** - One attention strategy at a time (simple)

**Key Improvements**:
- Files renamed: `book_embs.npy` → `book_subject_embeddings.npy`
- Variables renamed: `book_embs` → `book_subject_embeddings`
- Methods renamed: `get_als_embeddings()` → `get_als_factors()`
- Self-documenting code: immediately clear what each variable contains

**Timeline**: 4 weeks (big bang, no backward compat)

**Confidence**: High (clear contracts, layer-by-layer testing, rollback plan)

**Next Steps**: 
1. Review and approve plan
2. Start Week 1: Foundation + Infrastructure
3. Daily standups to track progress