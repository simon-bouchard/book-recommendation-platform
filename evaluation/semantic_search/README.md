# evaluation/semantic_search/README.md
"""
Documentation for multi-index semantic search comparison tool.
"""

# Semantic Search Evaluation

Compare multiple semantic search indexes (baseline, baseline_clean, v1_subjects, v1_full, v2_subjects, v2_full) to evaluate search quality across different enrichment strategies.

## Purpose

Determine the incremental value of LLM enrichment by comparing:
- **Baseline** → Raw metadata (title, author, description)
- **Baseline Clean** → Cleaned/normalized metadata
- **V1 Subjects** → Baseline + LLM-extracted subjects
- **V1 Full** → V1 Subjects + genre + tones + vibe
- **V2 Subjects** → Improved subject extraction (V2)
- **V2 Full** → V2 Subjects + improved genre/tones/vibe

## Quick Start

### 1. Prepare Test Queries

Edit `test_queries.json` to define your test queries. See [Query Types](#query-types) below.

### 2. Run Comparison

```bash
cd evaluation/semantic_search

# Compare all 6 indexes
python compare_indexes.py \
    --index baseline=~/bookrec/models/data/baseline \
    --index baseline_clean=~/bookrec/models/data/baseline_clean \
    --index v1_subjects=~/bookrec/models/data/enriched_v1_subjects \
    --index v1_full=~/bookrec/models/data/enriched_v1 \
    --index v2_subjects=~/bookrec/models/data/enriched_v2_subjects \
    --index v2_full=~/bookrec/models/data/enriched_v2 \
    --queries test_queries.json \
    --output results/

# Compare subset (e.g., just V1 variants)
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index v1_subjects=models/data/enriched_v1_subjects \
    --index v1_full=models/data/enriched_v1 \
    --output results/v1_comparison/

# Compare versions (full indexes only)
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index v1_full=models/data/enriched_v1 \
    --index v2_full=models/data/enriched_v2 \
    --output results/version_comparison/
```

### 3. Review Results

Output files are generated with timestamps:

```
results/
├── baseline_20241103_143045.json           # All results for baseline
├── baseline_clean_20241103_143045.json     # All results for baseline_clean
├── v1_subjects_20241103_143045.json        # All results for v1_subjects
├── v1_full_20241103_143045.json            # All results for v1_full
├── v2_subjects_20241103_143045.json        # All results for v2_subjects
├── v2_full_20241103_143045.json            # All results for v2_full
├── summary_20241103_143045.json            # Aggregate metrics
└── comparison_20241103_143045.html         # Visual side-by-side
```

## Query Types

### Exact Match (`exact_match`)

Programmatic validation - must return specific known books.

```json
{
  "id": 1,
  "text": "harry potter sorcerer's stone",
  "type": "exact_match",
  "description": "Direct title search",
  "complexity_level": "L1_exact_match",
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

**Validation:** Script checks if expected book appears in top K results for ALL indexes.

### Descriptive Queries (`descriptive`)

Manual review - 2-3 descriptive terms testing basic semantic understanding.

```json
{
  "id": 6,
  "text": "cozy mystery",
  "type": "descriptive",
  "complexity_level": "L2_two_descriptors",
  "description": "Two descriptors - tests basic semantic understanding",
  "manual_review": true
}
```

### Thematic Queries (`thematic`)

Manual review - complex multi-concept searches.

```json
{
  "id": 17,
  "text": "dystopian totalitarian surveillance mind control",
  "type": "thematic",
  "complexity_level": "L4_very_specific",
  "description": "Multi-concept dystopian themes - tests 1984-like books",
  "manual_review": true
}
```

### Complexity Levels

- **L1_exact_match** - Exact title/author searches (validation)
- **L2_two_descriptors** - Two descriptors (e.g., "dark fantasy")
- **L3_multi_descriptors** - 3-4 descriptors (e.g., "dark atmospheric gothic horror")
- **L4_very_specific** - Complex multi-concept queries (e.g., "Victorian detective solving murders foggy London")

## Output Files

### Individual Index JSONs (`{index_name}_{timestamp}.json`)

Complete results for each index:

```json
{
  "index_name": "v1_subjects",
  "queries": [
    {
      "query_id": 1,
      "query_text": "The Great Gatsby",
      "query_type": "exact_match",
      "complexity_level": "L1_exact_match",
      "description": "Classic American novel - exact title match",
      "results": [
        {
          "rank": 1,
          "book_id": 1452,
          "title": "The Great Gatsby",
          "author": "F. Scott Fitzgerald"
        },
        ...
      ],
      "assertion_results": [
        {
          "expected": {...},
          "found": true,
          "rank": 1,
          "matched_by": "item_idx"
        }
      ]
    },
    ...
  ]
}
```

**Best for:** LLM analysis, detailed per-index review.

### HTML Report (`comparison_{timestamp}.html`)

Visual side-by-side comparison with:
- Summary dashboard (pass rates per index)
- Exact match tests with color-coded pass/fail
- Quality queries for manual review (top 8 results per index)
- All indexes displayed in grid layout

**Best for:** Human review and presentation.

### Summary JSON (`summary_{timestamp}.json`)

High-level metrics across all indexes:

```json
{
  "metadata": {...},
  "indexes": ["baseline", "baseline_clean", "v1_subjects", ...],
  "total_queries": 20,
  "exact_match_results": {
    "baseline": {"passed": 4, "total": 5, "pass_rate": 0.8},
    "v1_subjects": {"passed": 5, "total": 5, "pass_rate": 1.0},
    ...
  },
  "query_type_breakdown": {
    "exact_match": 5,
    "descriptive": 10,
    "thematic": 5
  }
}
```

**Best for:** Quick decision making, dashboards.

## Metrics Explained

### Exact Match Pass Rates

- **Pass:** Expected book found in top K positions
- **Fail:** Expected book not found or ranked too low
- **Goal:** All indexes should pass >90% (basic relevance check)

### Pairwise Overlap

- **Overlap Top 5/10:** Number of books appearing in both indexes
- Calculated for all index pairs
- High overlap = indexes agree on relevance
- Low overlap = indexes find different books

### Author Diversity

- Number of unique authors in top 10 results
- Higher = more diverse recommendations
- Calculated per index per query

## Interpreting Results

### Exact Match Queries

Check if all indexes pass >90%. If not:
- Baseline failing → Index quality issues
- Only enriched failing → Enrichment hurting exact matches

### Quality Queries (Manual Review)

Evaluate based on:
1. **Relevance:** Do results match the query intent?
2. **Diversity:** Mix of authors, sub-genres, eras?
3. **Discovery:** Non-obvious but highly relevant books?
4. **Serendipity:** Interesting connections?

### Incremental Value Analysis

Compare results to understand what each enrichment adds:
- **Baseline → Baseline Clean:** Does cleaning improve results?
- **Baseline → V1 Subjects:** Does subject extraction help?
- **V1 Subjects → V1 Full:** Does genre/tones/vibe add value?
- **V1 → V2:** Does improved enrichment help?

## Example Workflow

```bash
# 1. Define test queries
vim test_queries.json

# 2. Run comparison on all indexes
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index baseline_clean=models/data/baseline_clean \
    --index v1_subjects=models/data/enriched_v1_subjects \
    --index v1_full=models/data/enriched_v1 \
    --index v2_subjects=models/data/enriched_v2_subjects \
    --index v2_full=models/data/enriched_v2 \
    --output results/

# 3. Open HTML report for human review
open results/comparison_20241103_143045.html

# 4. Review exact match pass rates
# - Should see >90% pass rate for all indexes
# - Note any failures

# 5. Manually review quality queries
# - Compare top 5-8 results across indexes
# - Note which index gives better results per query type

# 6. Send individual JSONs to LLM for analysis
# Upload baseline_*.json, v1_full_*.json, v2_full_*.json to Claude
# Prompt: "Analyze these semantic search results. Compare the quality 
#          of results across indexes. Which index performs best for:
#          - L2 queries (two descriptors)?
#          - L3 queries (multi descriptors)?
#          - L4 queries (very specific)?
#          Does V1 full justify the extra metadata vs V1 subjects?
#          Does V2 show improvement over V1?
#          Overall recommendation for production?"
```

## Adding New Queries

1. Choose appropriate query type and complexity level
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

- **L2 (two descriptors):** Basic mood/genre combos ("cozy mystery", "dark fantasy")
- **L3 (multi descriptors):** Multiple style elements ("dark atmospheric gothic horror")
- **L4 (very specific):** Complex scenarios ("Victorian detective solving murders foggy London")
- **Edge cases:** Unusual combinations, niche interests

### Increasing Test Coverage

Start with ~20 queries:
- 5 exact match (L1) - validation
- 5 descriptive (L2) - basic semantic understanding
- 5 multi-descriptor (L3) - complex descriptions
- 5 thematic (L4) - specific scenarios

Expand based on findings.

## Advanced Usage

### Compare Specific Subsets

```bash
# Just baseline variants
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index baseline_clean=models/data/baseline_clean

# Just subjects-only variants
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index v1_subjects=models/data/enriched_v1_subjects \
    --index v2_subjects=models/data/enriched_v2_subjects

# Just full variants
python compare_indexes.py \
    --index v1_full=models/data/enriched_v1 \
    --index v2_full=models/data/enriched_v2
```

### Custom Embedder

```bash
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index v1_full=models/data/enriched_v1 \
    --embedder sentence-transformers/all-mpnet-base-v2
```

**Note:** Must match the embedder used to build indexes.

### Higher Top K

```bash
python compare_indexes.py \
    --index baseline=models/data/baseline \
    --index v1_full=models/data/enriched_v1 \
    --top-k 20
```

Retrieves more results per query (useful for recall analysis).

## LLM Analysis Prompts

When sending results to Claude/ChatGPT, use prompts like:

**General Analysis:**
```
I've run semantic search comparisons across 6 indexes (baseline, baseline_clean, 
v1_subjects, v1_full, v2_subjects, v2_full). Analyze the attached JSON files.

For each query type (L2, L3, L4), which index performs best?
Does adding subjects help (baseline → v1_subjects)?
Does adding genre/tones/vibe help (v1_subjects → v1_full)?
Does V2 improve over V1?
What's your overall recommendation for production?
```

**Specific Focus:**
```
Focus on L4 (very specific) queries. Which index provides the most relevant 
and diverse results for complex, multi-concept searches?
```

**Incremental Value:**
```
Compare v1_subjects vs v1_full. Does the additional metadata (genre, tones, 
vibe) provide enough value to justify the extra enrichment complexity?
```

## Troubleshooting

### Import Errors

If `SemanticSearcher` import fails, script uses fallback implementation (automatic).

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

### Indexes Don't Load

```bash
# Verify index structure
ls -la models/data/enriched_v1/
# Should see: semantic.faiss, semantic_ids.npy, semantic_meta.json

# Check file sizes
du -h models/data/enriched_v1/*
```

## File Structure

```
evaluation/
├── __init__.py
├── .gitignore
└── semantic_search/
    ├── __init__.py
    ├── test_queries.json           # Test query definitions
    ├── compare_indexes.py          # Multi-index comparison script
    ├── example_commands.sh         # Usage examples
    ├── results/                    # Output directory (gitignored)
    │   ├── .gitkeep
    │   ├── baseline_*.json         # Per-index results
    │   ├── v1_full_*.json
    │   ├── v2_full_*.json
    │   ├── comparison_*.html       # Visual reports
    │   └── summary_*.json          # High-level metrics
    └── README.md                   # This file
```

## Dependencies

- `sentence-transformers` - Embedding model
- `faiss-cpu` or `faiss-gpu` - Vector search
- `scipy` - Statistical functions (optional, for advanced metrics)
- `numpy` - Numerical operations

Already installed if semantic search is in production.

## Next Steps

After running comparison:

1. **Review HTML report** - Quick visual overview of all indexes
2. **Analyze exact matches** - All indexes should pass >90%
3. **Manual review quality queries** - Which index gives best results per query type?
4. **LLM analysis** - Send JSONs to Claude/ChatGPT for detailed assessment
5. **Incremental analysis** - What does each enrichment add?
6. **Document findings** - Note strengths/weaknesses of each index
7. **Make decision** - Which index(es) to use in production?

## Questions?

- Check example queries in `test_queries.json`
- Review `example_commands.sh` for usage patterns
- Review output files for structure
- Inspect `compare_indexes.py` for implementation details
