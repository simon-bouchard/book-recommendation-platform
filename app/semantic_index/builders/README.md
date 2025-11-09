# Batch Index Builders

**Purpose**: One-time scripts to build FAISS indexes for the 6-model comparison experiment.

## What's Here

- `build_enriched_index.py` - Universal builder for v1/v2, full/subjects variants
- `build_baseline_clean_index.py` - Baseline without description variant
- `__init__.py` - Module marker

## Quick Start

### Rebuild V1-Full (Fix Bugs - PRIORITY!)

```bash
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --full \
  --output models/data/enriched_v1
```

### Build V1-Subjects

```bash
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --output models/data/enriched_v1_subjects
```

### Build Baseline-Clean

```bash
python -m app.semantic_index.builders.build_baseline_clean_index \
  --output models/data/baseline_clean
```

### Build V2 Variants (When V2 Enrichment ≥98%)

```bash
# V2-Subjects
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v2 \
  --output models/data/enriched_v2_subjects

# V2-Full
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v2 \
  --full \
  --output models/data/enriched_v2_full
```

## How It Works

### build_enriched_index.py (Universal Builder)

**Handles**: V1-Full, V1-Subjects, V2-Subjects, V2-Full

**Flags**:
- `--tags-version v1|v2` - Choose enrichment version
- `--full` - Include metadata (genre/tones/vibe), omit for subjects-only
- `--output DIR` - Output directory
- `--limit N` - Test with N books

**Process**:
1. Load ontology CSVs (tones, genres)
2. Query enriched books from SQL
3. Resolve IDs to names (tone_id → "dark", genre_id → "Science Fiction")
4. Build embedding text using templates
5. Embed with sentence-transformers
6. Build FAISS HNSW index
7. Save index + metadata

### build_baseline_clean_index.py

**Handles**: Baseline-Clean (no description)

**Process**:
1. Query books with OL subjects
2. Skip books without subjects
3. Build text: `{title} — {author} | subjects: {subjects}`
4. Embed and index
5. Save

## Output Structure

Each builder creates:
```
models/data/{variant}/
├── semantic.faiss        # FAISS HNSW index (~300MB)
├── semantic_ids.npy      # Book item_idx array (~2MB)
└── semantic_meta.json    # Metadata (title, author, etc.) (~50MB)
```

## Build Order

1. ✅ **V1-Full** (rebuild, fixes bugs) - ~2-3 hours
2. ✅ **V1-Subjects** - ~1.5-2 hours
3. ✅ **Baseline-Clean** - ~2-3 hours
4. ⏸️ **V2-Subjects** (wait for v2 ≥98%) - ~2-3 hours
5. ⏸️ **V2-Full** (wait for v2 ≥98%) - ~2-3 hours

## Test Mode

Always test with small limit first:

```bash
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --full \
  --limit 100 \
  --output test_output
```

## Troubleshooting

**"Tone ontology not found"**
- Verify `ontology/tones_v1.csv` and `ontology/tones_v2.csv` exist

**"No valid enriched books found"**
- Check database has data for that tags_version
- Verify enrichment tables populated

**Import errors**
- Ensure you're in project root
- Use `python -m app.semantic_index.builders.X` not `python app/semantic_index/builders/X.py`

## Differences from Stream Pipeline

| Aspect | Batch Builders | Stream Pipeline |
|--------|---------------|-----------------|
| Purpose | One-time index building | Continuous embedding |
| Data Source | SQL database | Kafka events |
| Output | FAISS index files | .npz accumulator batches |
| When to Use | Initial load, rebuilds | Production updates |
| Location | `builders/` | `embedding/` |

## See Also

- `app/semantic_index/templates/` - Text template functions
- `app/semantic_index/embedding/` - Continuous streaming pipeline
- `app/semantic_index/shared/` - Shared utilities
