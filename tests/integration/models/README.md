# Models Performance Testing Suite

This test suite measures the latency and performance of your recommendation and similarity models at the API level. Use it to establish a baseline before refactoring and compare results afterward.

## Overview

The suite tests:
- **Recommendation endpoints**: `/profile/recommend` with different user types
- **Similarity endpoints**: `/book/{item_idx}/similar` with different modes
- **Various parameters**: `top_n`, `top_k`, `alpha`, `mode`
- **Concurrent load**: Simulated concurrent requests
- **Cold start**: First request vs. cached requests

## Quick Start

### 1. Setup Test Data

First, identify suitable user and book IDs for testing:

```bash
# Generate test data configuration
python tests/integration/models/setup_test_data.py

# This creates: tests/integration/models/test_data_config.json
```

Review the generated `test_data_config.json` and copy the IDs into `test_models_performance.py`:

```python
# In test_models_performance.py
WARM_USER_IDS = [123, 456, 789, ...]  # Users with >=10 ratings
COLD_USER_IDS = [111, 222, 333, ...]  # Users with 1-9 ratings
NO_SUBJECT_USER_IDS = [444, 555, ...] # Users with no favorite subjects
TEST_BOOK_IDS = [1000, 2000, 3000, ...] # Books for similarity testing
```

### 2. Run Baseline Tests (Before Refactor)

```bash
# Run all performance tests
pytest tests/integration/models/test_models_performance.py -v -s

# Run with detailed output
pytest tests/integration/models/test_models_performance.py -v -s --tb=short

# Run specific test categories
pytest tests/integration/models/test_models_performance.py::test_warm_user_recommendations_latency -v
pytest tests/integration/models/test_models_performance.py::test_similar_books_latency -v
```

This will:
- Execute all performance tests
- Print a summary report at the end
- Save results to `tests/integration/models/performance_baselines/baseline_YYYYMMDD_HHMMSS.json`

### 3. Perform Your Refactoring

Make your changes to the models module...

### 4. Run Tests Again (After Refactor)

```bash
# Run the same tests again
pytest tests/integration/models/test_models_performance.py -v -s
```

### 5. Compare Results

```bash
# Auto-compare the two most recent baseline files
python tests/integration/models/compare_performance.py --auto

# Or specify files explicitly
python tests/integration/models/compare_performance.py baseline_20241216_140000.json baseline_20241216_150000.json

# Generate HTML report
python tests/integration/models/compare_performance.py --auto --html

# Fail CI/CD if regressions detected
python tests/integration/models/compare_performance.py --auto --fail-on-regression
```

## Test Categories

### User Recommendation Tests

#### Warm Users (>=10 ratings)
- **Test**: `test_warm_user_recommendations_latency`
- **Tests**: Auto mode, explicit behavioral mode
- **Expected**: Should use ALS-based recommendations
- **Typical latency**: 50-200ms

#### Cold Users (1-9 ratings)
- **Test**: `test_cold_user_recommendations_latency`
- **Tests**: Auto mode (routes to cold strategy)
- **Expected**: Should use subject-based recommendations
- **Typical latency**: 100-300ms

#### No Favorite Subjects
- **Test**: `test_no_subject_user_recommendations_latency`
- **Tests**: Fallback to Bayesian popularity
- **Expected**: Should return reasonable results
- **Typical latency**: 50-150ms

#### Varying top_n
- **Test**: `test_recommendations_varying_top_n`
- **Tests**: top_n = 50, 200, 500
- **Expected**: Linear scaling with top_n
- **Purpose**: Verify candidate generation doesn't become bottleneck

### Book Similarity Tests

#### Subject Mode
- **Test**: `test_similar_books_latency` with `mode="subject"`
- **Uses**: Attention-pooled subject embeddings + FAISS
- **Typical latency**: 10-50ms
- **Works for**: All books with subjects

#### ALS Mode
- **Test**: `test_similar_books_latency` with `mode="als"`
- **Uses**: Collaborative filtering factors
- **Typical latency**: 10-50ms
- **Works for**: Books with ALS data (min 10 ratings)

#### Hybrid Mode
- **Test**: `test_similar_books_latency` with `mode="hybrid"`
- **Uses**: Weighted combination of subject + ALS
- **Typical latency**: 20-100ms
- **Tests**: Various alpha values (0.3, 0.5, 0.7)

#### Varying top_k
- **Test**: `test_similarity_varying_top_k`
- **Tests**: top_k = 10, 50, 200, 500
- **Expected**: Minimal impact (FAISS is efficient)

### Stress Tests

#### Concurrent Requests
- **Test**: `test_concurrent_recommendation_requests`
- **Tests**: Sequential vs. 5 concurrent threads
- **Purpose**: Verify thread safety and resource contention
- **Expected**: <50% overhead from concurrency

#### Cold Start
- **Test**: `test_cold_start_latency`
- **Tests**: First request after model reload vs. subsequent requests
- **Purpose**: Measure model loading overhead
- **Expected**: First request 2-10x slower

## Configuration

### Environment Variables

```bash
# In your .env file
ADMIN_SECRET=your_secret_here  # For model reload endpoint
TESTING=1  # Enable test mode
```

### Test Parameters

Edit `test_models_performance.py` to adjust:

```python
# Number of warmup runs (discarded)
WARMUP_RUNS = 2

# Number of measurement runs
MEASUREMENT_RUNS = 10
```

## Understanding Results

### Latency Metrics

Each test measures:
- **Mean**: Average latency
- **Median**: 50th percentile (less affected by outliers)
- **P95**: 95th percentile (captures tail latency)
- **P99**: 99th percentile (worst case)
- **Min/Max**: Range of observed latencies
- **Stdev**: Consistency of performance

### What to Look For

**Good results:**
- Mean ≈ Median (consistent performance)
- Low stdev (predictable latency)
- P95 < 2x median (no extreme outliers)

**Potential issues:**
- Mean >> Median (outliers affecting average)
- High stdev (inconsistent performance)
- P95 >> Mean (tail latency problems)

### Acceptable Changes After Refactor

- **±5%**: Neutral (measurement noise)
- **-10% to -20%**: Good improvement
- **+10% to +20%**: Minor regression (investigate if systematic)
- **>+25%**: Significant regression (requires attention)

## Troubleshooting

### Tests Fail Due to Missing Data

**Problem**: Tests skip or fail because user/book IDs don't exist

**Solution**:
```bash
# Regenerate test data
python tests/integration/models/setup_test_data.py

# Verify the IDs
python tests/integration/models/setup_test_data.py --verify
```

### High Latency Variance

**Problem**: Stdev is high, results are inconsistent

**Possible causes**:
- Insufficient warmup runs
- System load during testing
- Database queries not cached
- Network issues

**Solutions**:
- Increase `WARMUP_RUNS` and `MEASUREMENT_RUNS`
- Run tests on idle system
- Run tests multiple times and compare

### Cold Start Too Slow

**Problem**: First request after model reload is very slow

**This is normal if**:
- Models are large (>100MB)
- Multiple models need loading
- FAISS indexes are large

**Consider**:
- Model preloading on startup
- Lazy loading optimization
- Smaller model sizes

### Concurrent Tests Show High Overhead

**Problem**: Concurrent requests are much slower than sequential

**Possible causes**:
- GIL (Global Interpreter Lock) contention
- Resource locking (database, model access)
- Memory thrashing

**Solutions**:
- Use multiprocessing instead of threading
- Implement connection pooling
- Profile with `py-spy` or similar tools

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Setup environment
        run: |
          # Install dependencies, setup database, etc.

      - name: Run baseline tests
        run: |
          pytest tests/integration/models/test_models_performance.py -v

      - name: Compare with main branch baseline
        run: |
          # Download baseline from main branch
          # Run comparison
          python tests/integration/models/compare_performance.py \
            baseline_main.json \
            baseline_pr.json \
            --fail-on-regression \
            --html

      - name: Upload results
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: performance-report
          path: tests/integration/models/performance_baselines/
```

## Best Practices

1. **Run tests on stable environment**: Avoid running during backups, deployments, or high load
2. **Multiple runs**: Run tests 3-5 times and average results for more reliable baselines
3. **Document system specs**: Include CPU, RAM, disk type in baseline metadata
4. **Version control baselines**: Commit baseline files to track performance over time
5. **Set thresholds**: Define acceptable regression thresholds for your use case
6. **Profile regressions**: Use `cProfile` or `py-spy` to investigate specific regressions
7. **Monitor production**: Use these tests to predict production performance

## Project Structure

```
tests/integration/models/
├── README.md                          # This file
├── test_models_performance.py         # Main test suite
├── setup_test_data.py                 # Test data configuration generator
├── compare_performance.py             # Comparison tool
├── test_data_config.json             # Generated test IDs (gitignore)
├── performance_baselines/            # Stored baseline results
│   ├── baseline_20241216_140000.json
│   ├── baseline_20241216_150000.json
│   └── comparison_20241216_151500.html
└── conftest.py                       # pytest fixtures (if needed)
```

## Example Output

```
PERFORMANCE TEST SUMMARY
================================================================================

recommend_warm_user_123_mode_auto:
  Mean: 87.23ms
  Median: 85.10ms
  P95: 105.30ms
  P99: 112.45ms
  Range: [78.20ms - 115.60ms]
  Stdev: 8.45ms
  Count: 10

similar_book_1000_mode_subject:
  Mean: 12.34ms
  Median: 11.90ms
  P95: 15.20ms
  P99: 16.10ms
  Range: [10.50ms - 17.20ms]
  Stdev: 1.85ms
  Count: 10

...
```

## Need Help?

- Check test output for detailed error messages
- Verify database connection and data availability
- Ensure models are properly loaded with `ModelStore`
- Check logs for warnings or errors during test execution
