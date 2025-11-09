# Batch Index Builders - WITH MULTIPROCESSING

**Purpose**: One-time scripts to build FAISS indexes for the 6-model comparison experiment.

**NEW**: 🚀 **Multi-process support** for 2-3x faster embedding on multi-core CPUs!

## What's Here

- `build_enriched_index.py` - Universal builder for v1/v2, full/subjects variants (WITH MULTIPROCESSING)
- `build_baseline_clean_index.py` - Baseline without description variant (WITH MULTIPROCESSING)
- `__init__.py` - Module marker

## 🚀 Quick Start (With Multiprocessing - RECOMMENDED)

### Rebuild V1-Full (Fix Bugs - PRIORITY!)

```bash
# Single-process (slow, ~2-3 hours)
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --full \
  --output models/data/enriched_v1

# Multi-process (FAST, ~1 hour with 4 cores) ⚡
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --full \
  --multiprocess \
  --num-processes 4 \
  --output models/data/enriched_v1
```

### Build V1-Subjects

```bash
# Multi-process (recommended)
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --multiprocess \
  --num-processes 4 \
  --output models/data/enriched_v1_subjects
```

### Build Baseline-Clean

```bash
# Multi-process (recommended)
python -m app.semantic_index.builders.build_baseline_clean_index \
  --multiprocess \
  --num-processes 4 \
  --output models/data/baseline_clean
```

### Build V2 Variants (When V2 Enrichment ≥98%)

```bash
# V2-Subjects (multi-process)
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v2 \
  --multiprocess \
  --num-processes 4 \
  --output models/data/enriched_v2_subjects

# V2-Full (multi-process)
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v2 \
  --full \
  --multiprocess \
  --num-processes 4 \
  --output models/data/enriched_v2_full
```

## ⚡ Multiprocessing Benefits

### Speed Comparison (250k books on 6-core machine)

| Mode | Cores Used | Time | Speedup |
|------|-----------|------|---------|
| Single-process | 1 | ~2-3 hours | 1x |
| Multi-process (2 workers) | 2 | ~1.5 hours | 2x |
| Multi-process (4 workers) | 4 | ~1 hour | 2.5-3x |
| Multi-process (6 workers) | 6 | ~50 min | 3x |

### When to Use Multiprocessing

✅ **Use `--multiprocess`** when:
- Running on CPU (not GPU)
- Have 4+ CPU cores available
- Processing large datasets (>10k books)
- Want faster builds

❌ **Don't use `--multiprocess`** when:
- Using GPU for embeddings
- Low memory (each process loads model)
- Testing with small datasets (`--limit 100`)

### Choosing `--num-processes`

**Rule of thumb**: Use `num_cores - 2` to leave cores for system

| Machine | Recommended |
|---------|-------------|
| 4-core | `--num-processes 2` |
| 6-core | `--num-processes 4` |
| 8-core | `--num-processes 6` |
| 12-core | `--num-processes 10` |

## All Flags

### build_enriched_index.py

```bash
--tags-version v1|v2        # Required: enrichment version
--full                      # Include genre/tones/vibe (omit for subjects-only)
--output DIR                # Required: output directory
--embedder MODEL            # Model name (default: all-MiniLM-L6-v2)
--multiprocess              # Enable multi-process encoding ⚡
--num-processes N           # Number of processes (default: 4)
--limit N                   # Test with N books
```

### build_baseline_clean_index.py

```bash
--output DIR                # Output directory (default: models/data/baseline_clean)
--embedder MODEL            # Model name (default: all-MiniLM-L6-v2)
--multiprocess              # Enable multi-process encoding ⚡
--num-processes N           # Number of processes (default: 4)
--limit N                   # Test with N books
```

## How It Works

### Multiprocessing Architecture

```
Main Process
    │
    ├─> Worker 1 (batch_size=32) → embeddings
    ├─> Worker 2 (batch_size=32) → embeddings
    ├─> Worker 3 (batch_size=32) → embeddings
    └─> Worker 4 (batch_size=32) → embeddings
         ↓
    Aggregate Results
         ↓
    Build FAISS Index
```

**Key settings**:
- Each worker processes batches of 32 texts
- Workers run in parallel on separate CPU cores
- Results aggregated after all workers complete
- Chunk size: 1000 texts per worker batch

### Single vs Multi-Process

**Single-process**:
```python
embeddings = model.encode(texts, batch_size=256)
```

**Multi-process**:
```python
pool = model.start_multi_process_pool(target_devices=['cpu'] * 4)
embeddings = model.encode_multi_process(texts, pool, batch_size=32)
model.stop_multi_process_pool(pool)
```

## Output Structure

Each builder creates:
```
models/data/{variant}/
├── semantic.faiss        # FAISS HNSW index (~300MB)
├── semantic_ids.npy      # Book item_idx array (~2MB)
└── semantic_meta.json    # Metadata (title, author, etc.) (~50MB)
```

## Build Order (WITH MULTIPROCESSING)

1. ✅ **V1-Full** (rebuild, fixes bugs) - ~1 hour (was 2-3h)
2. ✅ **V1-Subjects** - ~45 min (was 1.5-2h)
3. ✅ **Baseline-Clean** - ~1 hour (was 2-3h)
4. ⏸️ **V2-Subjects** (wait for v2 ≥98%) - ~1 hour
5. ⏸️ **V2-Full** (wait for v2 ≥98%) - ~1 hour

**Total time savings**: ~5-6 hours → ~2-3 hours (50% faster!)

## Test Mode

Always test with small limit first (single-process is fine for testing):

```bash
python -m app.semantic_index.builders.build_enriched_index \
  --tags-version v1 \
  --full \
  --limit 100 \
  --output test_output
```

## Monitoring Progress

Both modes show progress:

```
🚀 Using multi-process encoding (4 processes)...
Batches:  45%|███████████▌              | 450/1000 [02:15<02:45, 3.33batch/s]

✅ Created 235,550 embeddings (dim=384)
Building FAISS HNSW index...
✅ FAISS index built (235,550 vectors)
```

## Troubleshooting

### "Out of Memory" with Multiprocessing

**Problem**: Each process loads the model into memory
**Solution**: Reduce `--num-processes` (try 2 instead of 4)

### Multiprocessing Not Faster

**Problem**: Overhead on small datasets
**Solution**: Only use `--multiprocess` for 10k+ books

### CPU at 100% But Slow

**Problem**: Too many processes competing
**Solution**: Reduce `--num-processes` to `num_cores - 2`

## Performance Tips

### Optimal Settings for Your 6-Core Machine

```bash
# Best for 250k books
--multiprocess \
--num-processes 4

# Max speed (if memory allows)
--multiprocess \
--num-processes 6
```

### Memory Usage Estimate

- Single-process: ~2GB RAM
- Multi-process (4 workers): ~6-8GB RAM
- Multi-process (6 workers): ~10-12GB RAM

### Disk I/O

- Use SSD for faster model loading
- Output directory on fast disk
- SQLite database on SSD

## See Also

- `app/semantic_index/templates/` - Text template functions
- `app/semantic_index/embedding/` - Continuous streaming pipeline
- `app/semantic_index/shared/` - Shared utilities

---

**🎯 TLDR**: Add `--multiprocess --num-processes 4` to any builder command for 2-3x speedup!
