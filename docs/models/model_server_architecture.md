# Model Server Architecture Plan

## Core Principle

Service boundaries are drawn around **memory domains**, not business domains or
algorithms. Each service loads exactly the data its operations require. No array
or index is duplicated across services except where mathematically unavoidable.

---

## Model Servers

### 1. Embedder Service
**Owns:** PyTorch attention model weights only.

**Operation:** embed(subject_indices) → normalized query vector.

**Consumers:** unified similarity service (for subject recs), application layer.

---

### 2. Unified Similarity Service
**Owns:**
- Normalized subject embeddings (raw matrix, for matmul)
- HNSW subject index (for item-to-item queries)
- Normalized ALS book factors (raw matrix, for matmul)
- HNSW ALS index (for item-to-item queries)
- Bayesian scores (1D float array, negligible memory)
- Subject/ALS alignment map (row index mapping between the two spaces)

**Why unified:** hybrid similarity requires both the subject embedding matrix and
the ALS factor matrix in the same process simultaneously for a single-pass joint
matmul. Splitting them would require either duplicating both arrays across two
services or degrading to a two-list merge with lower recall. Since both arrays
must coexist anyway, subject sim, ALS sim, and subject recs come along for free
with no additional memory cost.

**Operations:**
- `subject_sim(item_idx, k)` — HNSW lookup using stored subject embedding
- `als_sim(item_idx, k)` — HNSW lookup using stored ALS factor
- `hybrid_sim(item_idx, k, alpha)` — single-pass joint matmul over both matrices
- `subject_recs(user_vector, k, subject_weight, popularity_weight)` — joint matmul
  over normalized subject embeddings and bayesian scores simultaneously

**Why subject_recs here and not in the application layer:**
Subject recommendations require a joint score of `cosine(user_vector, book_embeddings)
+ bayesian_score` computed over all books in one pass. Using FlatIndex or raw matmul
gives exact joint scoring. Using HNSW top-k followed by bayesian re-ranking would
lose recall because books with moderate cosine scores but high bayesian scores would
be cut from the initial retrieval pool before re-ranking could recover them.

The normalized subject embeddings are already in this service for hybrid sim, so
joint scoring is a free addition with no memory cost. Cosine on normalized vectors
is mathematically identical to the unnormalized matmul — no change in output.

---

### 3. ALS Recommendation Service
**Owns:** Non-normalized user factors and book factors (raw matrices).

**Why separate from the similarity service:** ALS recommendations use a raw dot
product (`book_factors @ user_vector`) where the magnitude of the factors carries
meaning — it represents predicted interaction strength. Normalizing the vectors
before this computation changes the semantics and degrades recommendation quality.
The similarity service stores normalized ALS factors for cosine similarity, which
is a different mathematical operation serving a different purpose. These are two
distinct representations of the same source data and cannot be shared.

**Operations:**
- `has_user(user_id)` — warm/cold user gate
- `als_recs(user_id, k)` — raw dot product over book factors

---

### 4. Metadata Service
**Owns:** Book metadata DataFrame, bayesian scores, subject TF-IDF index.

Note: bayesian scores are duplicated between this service (for enrichment and
popular books queries) and the unified similarity service (for joint scoring).
This is acceptable because the array is tiny relative to the embedding matrices.

**Operations:**
- `enrich(item_indices)` — return book metadata for a list of item_idx
- `popular(k)` — top-k books by bayesian score
- `subject_search(phrases)` — resolve subject phrase strings to subject_idx candidates

---

## The Unavoidable Duplication

ALS book factors exist in two forms:
- **Normalized** in the unified similarity service (cosine similarity)
- **Non-normalized** in the ALS recommendation service (dot product)

This is a mathematical requirement, not an architectural choice. The two forms
serve fundamentally different operations and cannot be unified without degrading
one of them.

Bayesian scores exist in both the unified similarity service and the metadata
service. This is acceptable given their negligible size.

---

## Application Layer (Stateless)

Sits above the model servers. Owns no arrays or indices. Handles:
- Hybrid blending logic for subject recs (calling embedder then similarity service)
- ALS recommendation pipeline (read filters, GBT re-ranking)
- Enrichment (calling metadata service after candidate generation)
- Business rules (warm/cold gating, mode selection)
- User-facing API surface

Whether this is a single monolith or multiple microservices is an independent
decision that does not affect memory management at all.

---

## Reload Coordination

Subject embeddings and the HNSW subject index always retrain together. Both live
in the unified similarity service, so reload is intra-service with no cross-service
coordination needed for the subject domain.

ALS factors retrain together with the HNSW ALS index. The similarity service
(normalized factors + HNSW ALS) and the ALS recommendation service
(non-normalized factors) both need to reload when ALS retrains. This is the one
case requiring coordinated reload across two services.

The coordinator calls each service's reload endpoint in sequence, gating on the
readiness check of the first before triggering the second.
