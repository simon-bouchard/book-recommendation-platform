# tests/integration/models/README.md
"""
Performance testing suite for recommendation and similarity models.
Establishes baseline metrics before refactoring and validates performance after changes.
"""

# Models Performance Testing Suite

This test suite measures the latency and performance of your recommendation and similarity models at the API level. Use it to establish a baseline before refactoring and compare results afterward.

## Overview

The suite tests all performance-critical code paths:
- Recommendation endpoints with different user types and modes
- Similarity endpoints with different modes and parameters
- Concurrent request handling
- Cold start vs cached performance

## Quick Start

### Prerequisites

The full local stack from the root README's Local Setup: database populated, artifacts promoted, and model servers running (`EMBEDDER_URL`, `SIMILARITY_URL`, `ALS_URL`, `METADATA_URL`, defaulting to `localhost:8001-8004`). The session waits up to 60s for each server's `/health` before failing. The app runs in-process, so no backend server is needed. The `train` extra isn't needed.

### 1. Generate Test Data

```bash
uv run python tests/integration/models/setup_test_data.py
```

This queries the database and the active ALS artifacts and writes `test_data_config.json`, which the test module reads at import time:

| Key | Selection (up to 10 each) | Path exercised |
|-----|---------------------------|----------------|
| `warm_user_ids` | >= 10 ratings, most-rated first | ALS |
| `cold_with_subjects_user_ids` | 1-9 ratings, has favorite subjects | Subject similarity + Bayesian blend |
| `cold_without_subjects_user_ids` | 1-9 ratings, no favorite subjects | Bayesian popularity fallback |
| `test_book_ids.popular` | >= 50 ratings | ALS / hybrid similarity |
| `test_book_ids.has_als` | Popular books present in the ALS model | ALS similarity |
| `test_book_ids.niche` | 5-15 ratings | Subject similarity |
| `test_book_ids.all` | Union of the three book groups | Used by the similarity tests |

The file is gitignored and specific to your database. Regenerate it after reimporting data or retraining ALS, and check an existing one with `--verify`. If it's missing, the suite is skipped with a message pointing here.

`tests/integration/model_servers/_utils.py` hardcodes the same IDs so results from both layers are comparable. Update it if you regenerate the config.

### 2. Run a Baseline (on `master`)

```bash
uv run pytest tests/integration/models/test_models_performance.py -v -s
```

Results are saved to `performance_baselines/baseline_YYYYMMDD_HHMMSS.json`.

### 3. Run Again on Your Branch

Same command, after making your changes.

### 4. Compare Results

```bash
uv run python tests/integration/models/compare_performance.py --auto
```

`--auto` compares the two most recent baselines. Pass two filenames instead to compare specific runs. `--detailed` shows a per-test breakdown, and `--fail-on-regression` exits non-zero if any group regresses.

## Test Categories

### User Recommendation Tests

#### Warm Users (>=10 ratings) - ALS Path
- **Test**: `test_warm_user_recommendations_latency`
- **Modes**: auto, behavioral
- **Code Path**: WarmRecommender -> ALSCandidateGenerator
- **Expected Latency**: 50-200ms
- **What's Tested**: ALS matrix multiplication performance

#### Warm Users Forced to Subject Mode
- **Test**: `test_warm_user_forced_subject_mode_latency`
- **Mode**: subject (explicit)
- **Code Path**: ColdRecommender -> ColdHybridCandidateGenerator
- **Expected Latency**: 100-300ms
- **What's Tested**: Subject similarity with large user history

#### Cold Users WITH Subjects - Hybrid Path
- **Test**: `test_cold_user_with_subjects_latency`
- **Code Path**: ColdRecommender -> ColdHybridCandidateGenerator (use_only_bayesian=False)
- **Expected Latency**: 100-300ms
- **What's Tested**: Attention pooling + similarity + Bayesian blending

#### Cold Users WITHOUT Subjects - Bayesian Fallback
- **Test**: `test_cold_user_without_subjects_latency`
- **Code Path**: ColdRecommender -> ColdHybridCandidateGenerator (use_only_bayesian=True)
- **Expected Latency**: 50-150ms (fastest cold path)
- **What's Tested**: Pure Bayesian popularity (no embeddings/similarity)

#### Varying w Parameter
- **Test**: `test_cold_recommendations_varying_w`
- **Parameters**: w=0.3, 0.6, 0.9
- **What's Tested**: Similarity vs Bayesian weight balance

#### Varying top_n
- **Test**: `test_recommendations_varying_top_n`
- **Parameters**: top_n=50, 200, 500
- **What's Tested**: Candidate generation scaling

### Book Similarity Tests

#### Subject Mode
- **Test**: `test_similar_books_latency` with mode="subject"
- **Code Path**: SubjectSimilarityStrategy -> FAISS IndexFlatIP
- **Expected Latency**: 10-50ms
- **Works For**: All books with subjects

#### ALS Mode
- **Test**: `test_similar_books_latency` with mode="als"
- **Code Path**: ALSSimilarityStrategy -> FAISS IndexFlatIP
- **Expected Latency**: 10-50ms
- **Works For**: Books in ALS training set (check via ModelStore().has_book_als())

#### Hybrid Mode
- **Test**: `test_similar_books_latency` with mode="hybrid"
- **Code Path**: HybridSimilarityStrategy (blends subject + ALS)
- **Expected Latency**: 20-100ms
- **Works For**: Books with subjects

#### Varying Alpha
- **Test**: `test_hybrid_similarity_varying_alpha`
- **Parameters**: alpha=0.3, 0.5, 0.7
- **What's Tested**: Subject vs ALS balance

#### Varying top_k
- **Test**: `test_similarity_varying_top_k`
- **Parameters**: top_k=10, 50, 200, 500
- **What's Tested**: FAISS search scaling

### Stress Tests

#### Concurrent Requests
- **Test**: `test_concurrent_recommendation_requests`
- **What's Tested**: Sequential vs 5 concurrent threads
- **Expected**: <50% overhead from concurrency

#### Cold Start
- **Test**: `test_cold_start_latency`
- **What's Tested**: First request after model reload vs subsequent
- **Expected**: 2-10x slower for first request

## Configuration

`conftest.py` sets `TESTING`, `SECURE_MODE=false` and a `TEST_RUN_ID` for tracing. It also turns off the route caching decorators so the numbers show real pipeline latency, not cache hits.

### Test Parameters

Edit `test_models_performance.py`:

```python
WARMUP_RUNS = 10  # Discarded measurements
MEASUREMENT_RUNS = 50  # Actual measurements
```

## Understanding Results

### Latency Metrics

Each test measures:
- **Mean**: Average latency
- **Median**: 50th percentile
- **P95**: 95th percentile (captures tail latency)
- **P99**: 99th percentile (worst case)
- **Min/Max**: Range of observed latencies
- **Stdev**: Performance consistency

### What to Look For

**Good results:**
- Mean ≈ Median (consistent performance)
- Low stdev (predictable latency)
- P95 < 2x median

**Potential issues:**
- Mean >> Median (outliers)
- High stdev (inconsistent)
- P95 >> Mean (tail latency problems)

### Acceptable Changes After Refactor

- **±5%**: Neutral (measurement noise)
- **-10% to -20%**: Good improvement
- **+10% to +20%**: Minor regression (investigate if systematic)
- **>+25%**: Significant regression (requires attention)

## Critical Performance Paths

### Most Important to Measure

1. **Cold without subjects** (Bayesian fallback)
   - Fastest cold path
   - No embedding/similarity computation
   - Should be significantly faster than cold-with-subjects

2. **Warm user ALS**
   - Most common production path for engaged users
   - Matrix multiplication performance critical

3. **Cold with subjects**
   - New user with preferences
   - Attention pooling + similarity computation

### Less Critical But Still Measured

4. **Warm forced subject mode**
   - Edge case but valid API call
   - Tests subject similarity scaling

5. **Parameter variations** (w, alpha, top_n, top_k)
   - Ensures no unexpected scaling issues

## Best Practices

1. Run tests on stable environment (avoid during high load)
2. Multiple runs (3-5 times) for reliable baselines
3. Document system specs in baseline metadata
4. Version control baseline files
5. Set acceptable regression thresholds for your use case
6. Profile specific regressions with cProfile or py-spy
7. Monitor production performance alongside tests

## Troubleshooting

### Suite Skipped: test_data_config.json Not Found

Generate it (see Quick Start step 1):
```bash
uv run python tests/integration/models/setup_test_data.py
```

### High Latency Variance

Increase `WARMUP_RUNS` and `MEASUREMENT_RUNS`. Run on idle system.

### Cold Start Too Slow

This is normal for large models. Consider:
- Model preloading on startup
- Lazy loading optimization
- Smaller model sizes

### Concurrent Tests Show High Overhead

Possible causes:
- GIL contention
- Database connection pooling issues
- Resource locking

Consider using multiprocessing instead of threading.

## Project Structure

```
tests/integration/models/
├── README.md
├── test_models_performance.py
├── setup_test_data.py
├── compare_performance.py
├── analyze_baseline.py
├── conftest.py
├── test_data_config.json (generated, gitignored)
└── performance_baselines/
    └── baseline_YYYYMMDD_HHMMSS.json
```
