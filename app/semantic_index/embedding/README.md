# Phase 3: Continuous Embedding Generation

## Overview

This module implements the continuous embedding generation system for enriched books. It generates semantic embeddings from enriched metadata (subjects, tones, genres, vibes) and stores them incrementally for later FAISS index building.

## Components

1. **OntologyResolver** - Loads tone/genre mappings from CSV files
2. **FingerprintTracker** - SQLite-based deduplication (avoids re-embedding unchanged items)
3. **EnrichmentFetcher** - Queries enriched books from MySQL with multi-table joins
4. **EmbeddingClient** - Wrapper for sentence-transformers/all-MiniLM-L6-v2
5. **AccumulatorWriter** - Writes embeddings to NPZ batch files
6. **CoverageMonitor** - Tracks progress in metadata.json
7. **EmbeddingWorker** - Main orchestration loop

## Directory Structure

```
models/data/enriched_v2/
├── accumulator/              # NPZ batch files
│   ├── batch_000001.npz
│   ├── batch_000002.npz
│   └── ...
├── fingerprints.db          # SQLite fingerprint tracker
└── metadata.json            # Coverage statistics
```

## Usage

### Test Run (10 items)
```bash
cd /home/simon/documents/book_recsys

python -m app.semantic_index.embedding.worker \
    --mode full \
    --tags-version v2 \
    --limit 10 \
    --batch-size 10 \
    --output-dir models/data/enriched_v2
```

### Test Run (100 items)
```bash
python -m app.semantic_index.embedding.worker \
    --mode full \
    --tags-version v2 \
    --limit 100 \
    --batch-size 64
```

### Test Run (1000 items)
```bash
python -m app.semantic_index.embedding.worker \
    --mode full \
    --tags-version v2 \
    --limit 1000 \
    --batch-size 128
```

### Full Initial Load (~170k items)
```bash
python -m app.semantic_index.embedding.worker \
    --mode full \
    --tags-version v2 \
    --batch-size 128 \
    --quality-tiers RICH SPARSE MINIMAL BASIC
```

### Incremental Run (continuous updates)
```bash
# Run every 15-30 minutes to pick up new enrichments
python -m app.semantic_index.embedding.worker \
    --mode incremental \
    --tags-version v2 \
    --limit 1000 \
    --batch-size 128
```

## CLI Arguments

- `--mode`: `full` (initial load) or `incremental` (updates only)
- `--tags-version`: Tags version to process (default: `v2`)
- `--batch-size`: Embedding batch size (default: `128`)
- `--limit`: Max items to process (default: all)
- `--output-dir`: Output directory (default: `models/data/enriched_v2`)
- `--ontology-dir`: Ontology CSV directory (default: `ontology`)
- `--device`: Device for model (`cpu`, `cuda`, `mps`) (default: `cpu`)
- `--quality-tiers`: Quality tiers to include (default: `RICH SPARSE MINIMAL BASIC`)

## How It Works

### Full Load Mode
1. Fetches all enriched items from MySQL (filtered by tags_version)
2. For each batch:
   - Checks fingerprints (skip if unchanged)
   - Resolves tone IDs → names, genre slug → name
   - Builds embedding text: `"{title} — {author} | subjects: {subjects} | tones: {tones} | genre: {genre} | vibe: {vibe}"`
   - Generates embeddings (384-dim vectors)
   - Writes to NPZ batch file
   - Updates fingerprints and coverage stats

### Incremental Mode
1. Fetches recent items from MySQL
2. Filters by fingerprint (only changed items)
3. Processes only new/modified items
4. Updates coverage stats

### Fingerprint System
- Fingerprint = SHA256 hash of (item_idx, title, author, subjects, tone_ids, genre, vibe, tags_version)
- Only re-embeds if content changed
- Stored in SQLite for fast lookups

## Output Files

### NPZ Batch Files
Each `batch_XXXXXX.npz` contains:
- `embeddings`: (N, 384) float32 array
- `item_indices`: (N,) int64 array of item_idx values
- `metadata`: JSON string with enrichment data

### metadata.json
```json
{
  "tags_version": "v2",
  "embed_version": "emb_v1",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "total_items": 170000,
  "embedded_items": 150000,
  "coverage_percent": 88.24,
  "created_at": "2024-11-05T10:00:00",
  "last_updated": "2024-11-05T15:30:00",
  "runs": [...],
  "errors": [...]
}
```

## Next Steps: Building FAISS Index

After embedding is complete (or reaches sufficient coverage), build the FAISS index:

```python
from app.semantic_index.embedding import AccumulatorWriter

# Consolidate all batches
writer = AccumulatorWriter("models/data/enriched_v2")
embeddings, item_indices, metadata = writer.consolidate_batches()

# Build FAISS index (use existing build_index.py logic)
# ... FAISS building code ...
```

## Performance Notes

- **Batch size 128**: Good balance for 6 vCPU server
- **Expected throughput**: ~1000-2000 items/hour (depends on CPU)
- **Full load (170k)**: ~85-170 hours sequential, can parallelize with multiple workers
- **Memory usage**: ~2-3GB per worker
- **Incremental runs**: <1 minute if <100 new items

## Monitoring

Check coverage anytime:
```bash
python -c "
from app.semantic_index.embedding import CoverageMonitor
monitor = CoverageMonitor('models/data/enriched_v2')
monitor.print_stats()
"
```

## Troubleshooting

**Error: "Tones CSV not found"**
- Ensure `ontology/tones_v2.csv` exists
- Check `--ontology-dir` argument

**Error: "No valid enrichment records found"**
- Verify enrichment data exists in database
- Check `--tags-version` matches your enrichment version

**Database connection errors**
- Verify `.env` has correct `DATABASE_URL`
- Check MySQL is running and accessible

**Out of memory**
- Reduce `--batch-size` to 64 or 32
- Process in smaller chunks with `--limit`
