# Semantic Search Evaluation

Compare baseline (raw metadata) vs enriched (LLM-enhanced) semantic search indexes to evaluate the impact of LLM enrichment on search quality.

## Purpose

Determine whether LLM enrichment (subjects, tones, genre, vibe) provides enough search quality improvement to justify the streaming infrastructure investment.

## Quick Start

### 1. Prepare Test Queries

Edit `test_queries.json` to define your test queries. See [Query Types](#query-types) below.

### 2. Run Comparison

```bash
cd evaluation/semantic_search

# Basic usage (uses default paths)
python compare_indexes.py

# Custom paths
python compare_indexes.py \
    --baseline ~/bookrec/models/data/baseline \
    --enriched ~/bookrec/models/data/enriched_v1 \
    --queries test_queries.json \
    --output results/ \
    --top-k 10
```

### 3. Review Results

Three output files are generated with timestamps:

- **`comparison_YYYYMMDD_HHMMSS.html`** - Visual report (open in browser)
- **`comparison_YYYYMMDD_HHMMSS.json`** - Full structured results
- **`summary_YYYYMMDD_HHMMSS.json`** - High-level metrics and recommendation

## Query Types

### Exact Match (`exact_match`)

Programmatic validation - must return specific known books.

```json
{
  "id": 1,
  "text": "harry potter sorcerer's stone",
  "type": "exact_match",
  "description": "Direct title search",
  "expected_items": [
    {
      "title": "Harry Potter and the Sorcerer's Stone",
      "author": "J.K. Rowling",
      "item_idx": 12345,
      "must_appear_in_top": 3
    }
  ]
}
```

**Validation:** Script checks if expected book appears in top K results.

### Vibe Queries (`vibe`)

Manual review - mood/tone-based searches.

```json
{
  "id": 3,
  "text": "dark atmospheric mystery",
  "type": "vibe",
  "description": "Tests tone enrichment",
  "expected_better": "enriched",
  "manual_review": true
}
```

### Thematic Queries (`thematic`)

Manual review - subject/concept searches.

```json
{
  "id": 4,
  "text": "dystopian surveillance states",
  "type": "thematic",
  "description": "Tests subject enrichment",
  "manual_review": true
}
```

### Genre Queries (`genre`)

Manual review - genre-specific searches.

```json
{
  "id": 5,
  "text": "hard science fiction space opera",
  "type": "genre",
  "description": "Tests genre classification",
  "manual_review": true
}
```

## Output Files

### HTML Report

Visual side-by-side comparison with:
- Summary dashboard (pass rates, score improvements, overlap)
- Exact match tests with color-coded pass/fail
- Quality queries for manual review
- Top results for each query

**Best for:** Human review and presentation.

### Full JSON (`comparison_*.json`)

Complete structured results including:
- All query results with full rankings
- Detailed metrics (overlap, scores, correlations)
- Assertion details for exact match queries
- Summary statistics

**Best for:** Programmatic analysis, archiving, LLM review.

### Summary JSON (`summary_*.json`)

High-level metrics only:
- Pass rates by index
- Score improvements
- Recommendation (baseline/enriched)
- Confidence level

**Best for:** Quick decision making, dashboards.

## Metrics Explained

### Overlap Metrics

- **Overlap Top 5/10:** Number of books appearing in both result sets
- High overlap = indexes agree on relevance
- Low overlap = indexes find different books

### Score Metrics

- **Average Score:** Mean similarity score in top 10 results
- **Improvement:** Difference between enriched and baseline
- Higher scores = more confident matches

### Rank Correlation

- **Spearman's Rho:** How similarly indexes rank common books
- 1.0 = perfect agreement, 0.0 = no correlation
- Only calculated for books appearing in both results

### Author Diversity

- Number of unique authors in top 10 results
- Higher = more diverse recommendations

## Interpreting Results

### Exact Match Queries

**Pass:** Expected book found in top K positions
**Fail:** Expected book not found or ranked too low

**Goal:** Both indexes should pass >90% of exact matches (basic relevance).

### Quality Queries (Manual Review)

Evaluate based on:
1. **Relevance:** Do results match the query intent?
2. **Diversity:** Mix of authors, sub-genres, eras?
3. **Discovery:** Non-obvious but highly relevant books?
4. **Serendipity:** Interesting connections?

### Score Improvements

- **+10-20%:** Significant improvement
- **0-10%:** Modest improvement
- **Negative:** Baseline may be better for some queries

## Example Workflow

```bash
# 1. Define test queries
vim test_queries.json

# 2. Run comparison
python compare_indexes.py --output results/

# 3. Open HTML report
open results/comparison_20241103_153045.html

# 4. Review exact match results
# - Should see high pass rates (>90%) for both indexes
# - Note any failures

# 5. Manually review quality queries
# - Compare top 5-8 results for each
# - Note which index gives better results

# 6. (Optional) Send JSON to LLM for analysis
# Upload comparison JSON to Claude with prompt:
# "Analyze these semantic search results. Which index performs better
#  for vibe queries? For thematic queries? Overall recommendation?"
```

## Adding New Queries

1. Choose appropriate query type
2. Add to `test_queries.json`
3. For exact_match: specify expected books (title, author, optional item_idx)
4. For quality: mark as `manual_review: true`
5. Re-run comparison

## Tips

### Finding Good Exact Match Queries

- Use well-known books with distinctive titles
- Include author name to avoid ambiguity
- Test both popular and niche books
- Verify expected item_idx by searching production index

### Designing Quality Queries

- **Vibe:** Focus on mood/atmosphere (e.g., "cozy mystery", "grimdark fantasy")
- **Thematic:** Multi-concept queries (e.g., "AI consciousness ethics")
- **Genre:** Specific sub-genres (e.g., "Regency romance", "cyberpunk noir")
- **Edge cases:** Unusual combinations, niche interests

### Increasing Test Coverage

Start with ~15-20 queries:
- 5-8 exact match (validation)
- 5-8 vibe queries (tone enrichment)
- 5-8 thematic queries (subject enrichment)

Expand based on findings.

## Troubleshooting

### Import Errors

If `SemanticSearcher` import fails, script uses fallback implementation.

### Missing Indexes

Ensure paths point to directories containing:
- `semantic.faiss`
- `semantic_ids.npy`
- `semantic_meta.json`

### No Results

- Check if queries match any content in indexes
- Try broader queries
- Verify embedding model matches index

### Low Pass Rates

- Review expected_items (title/author spelling)
- Check item_idx values
- Try increasing `must_appear_in_top` threshold

## Advanced Usage

### Custom Embedder

```bash
python compare_indexes.py \
    --embedder sentence-transformers/all-mpnet-base-v2
```

**Note:** Must match the embedder used to build indexes.

### Higher Top K

```bash
python compare_indexes.py --top-k 20
```

Retrieves more results per query (useful for recall analysis).

### Batch Comparisons

```bash
# Compare multiple query sets
for queries in queries_*.json; do
    python compare_indexes.py --queries $queries --output results/
done
```

## File Structure

```
evaluation/
├── __init__.py
├── .gitignore
└── semantic_search/
    ├── __init__.py
    ├── test_queries.json       # Test query definitions
    ├── compare_indexes.py      # Main comparison script
    ├── results/                # Output directory (gitignored)
    │   ├── .gitkeep
    │   ├── comparison_*.json   # Full results
    │   ├── comparison_*.html   # Visual reports
    │   └── summary_*.json      # High-level metrics
    └── README.md               # This file
```

## Dependencies

- `sentence-transformers` - Embedding model
- `faiss-cpu` or `faiss-gpu` - Vector search
- `scipy` - Statistical functions (rank correlation)
- `numpy` - Numerical operations

Already installed if semantic search is in production.

## Next Steps

After running comparison:

1. **Review HTML report** - Quick visual overview
2. **Analyze exact matches** - Both indexes should pass >90%
3. **Manual review quality queries** - Which index gives better results?
4. **Optional: LLM analysis** - Send JSON to Claude for detailed assessment
5. **Document findings** - Note strengths/weaknesses of each index
6. **Make decision** - Does enrichment justify infrastructure?

## Questions?

- Check example queries in `test_queries.json`
- Review output files for structure
- Inspect `compare_indexes.py` for implementation details
