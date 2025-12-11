# Quick Reference: Artifact Naming Changes

## Summary of Changes

This document provides a quick reference for the new naming conventions introduced in the models module refactor.

---

## 🎯 Core Principle

**Pattern**: `{entity}_{representation_type}_{optional_variant}.{ext}`

This makes it immediately clear:
- **What** the artifact contains (entity)
- **How** it was generated (representation type)
- **Which variant** if there are multiple (optional)

---

## 📦 Complete Rename Map

### Embeddings Directory

```
OLD NAME                    →  NEW NAME                          REASON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
book_embs.npy              →  book_subject_embeddings.npy       Clarifies these are subject-pooled
book_ids.json              →  book_subject_ids.json             Matches embedding file
book_als_emb.npy           →  book_als_factors.npy              "factors" more accurate for ALS
book_als_ids.json          →  book_als_ids.json                 ✓ Keep (already clear)
user_als_emb.npy           →  user_als_factors.npy              Consistent with book ALS
user_als_ids.json          →  user_als_ids.json                 ✓ Keep (already clear)
```

### Attention Directory

```
OLD NAME                                     →  NEW NAME                              REASON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
subject_attention_components.pth            →  subject_attention_scalar.pth          Shorter, clearer variant
subject_attention_components_perdim.pth     →  subject_attention_perdim.pth          Remove redundant "components"
subject_attention_components_selfattn.pth   →  subject_attention_selfattn.pth        Remove redundant "components"
subject_attention_components_selfattn_perdim.pth → subject_attention_selfattn_perdim.pth  Remove redundant "components"
```

### Scoring Directory

```
OLD NAME               →  NEW NAME                REASON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
bayesian_tensor.npy   →  bayesian_scores.npy     More descriptive of contents
gbt_cold.pickle       →  gbt_cold.pickle         ✓ Keep (already clear)
gbt_warm.pickle       →  gbt_warm.pickle         ✓ Keep (already clear)
```

---

## 📁 New Directory Structure

```
models/
├── artifacts/                          # NEW: All trained artifacts here
│   ├── embeddings/                     # Vector representations
│   │   ├── book_subject_embeddings.npy
│   │   ├── book_subject_ids.json
│   │   ├── book_als_factors.npy
│   │   ├── book_als_ids.json
│   │   ├── user_als_factors.npy
│   │   └── user_als_ids.json
│   ├── attention/                      # Attention pooling components
│   │   ├── subject_attention_scalar.pth
│   │   ├── subject_attention_perdim.pth
│   │   ├── subject_attention_selfattn.pth
│   │   └── subject_attention_selfattn_perdim.pth
│   └── scoring/                        # Ranking/scoring models
│       ├── bayesian_scores.npy
│       ├── gbt_cold.pickle
│       └── gbt_warm.pickle
├── core/                               # NEW: Foundation layer
│   ├── constants.py                    # PAD_IDX, etc.
│   ├── paths.py                        # Centralized path definitions
│   └── config.py                       # Optional env config
├── data/                               # NEW: Data loading layer
│   ├── loaders.py                      # ModelStore, load functions
│   └── queries.py                      # DB/DataFrame queries
└── training/
    ├── data/                           # Training input data (unchanged)
    └── *.py                            # Training scripts (updated)
```

---

## 🔍 Naming Rationale

### Why "subject_embeddings" not "embeddings"?

**Before**: `book_embs.npy` - ambiguous, which embeddings?
**After**: `book_subject_embeddings.npy` - clear, these are derived from subjects

The system now has multiple book representations:
- **Subject embeddings**: Attention-pooled from book subjects
- **ALS factors**: Collaborative filtering latent factors
- **Semantic embeddings**: LLM-generated (in separate index)

Each needs a distinct, descriptive name.

### Why "factors" not "embeddings" for ALS?

**Technical accuracy**: ALS produces *latent factors* from matrix factorization, not embeddings from text/features.

**Common terminology**: ML literature uses "factors" for factorization models.

### Why "scores" not "tensor"?

**Clarity**: These are precomputed Bayesian popularity scores, not a generic tensor.

**Convention**: Score arrays are commonly named `*_scores.npy` in ML systems.

### Why remove "components" from attention files?

**Redundancy**: The file already says "subject_attention" - no need for "components".

**Brevity**: `subject_attention_scalar.pth` is clearer than `subject_attention_components_scalar.pth`

---

## 💻 Code Migration Patterns

### Loading Embeddings (Old vs New)

```python
# OLD WAY
book_embs = np.load("models/data/book_embs.npy")
with open("models/data/book_ids.json") as f:
    book_ids = json.load(f)

# NEW WAY
from models.core.paths import PATHS
book_embs = np.load(PATHS.book_subject_embeddings)
with open(PATHS.book_subject_ids) as f:
    book_ids = json.load(f)

# BETTER: Use loader function
from models.data.loaders import load_book_subject_embeddings
book_embs, book_ids = load_book_subject_embeddings()
```

### Loading ALS (Old vs New)

```python
# OLD WAY
user_embs = np.load("models/data/user_als_emb.npy")
book_embs = np.load("models/data/book_als_emb.npy")

# NEW WAY
from models.core.paths import PATHS
user_factors = np.load(PATHS.user_als_factors)
book_factors = np.load(PATHS.book_als_factors)

# BETTER: Use loader function
from models.data.loaders import load_als_embeddings
user_factors, book_factors, user_ids, book_ids = load_als_embeddings()
```

### Loading Attention Models (Old vs New)

```python
# OLD WAY
path = f"models/data/subject_attention_components_{strategy}.pth"
state = torch.load(path)

# NEW WAY
from models.core.paths import PATHS
path = PATHS.get_attention_path(strategy)
state = torch.load(path)

# BETTER: Use ModelStore
from models.data.loaders import ModelStore
strategy = ModelStore().get_attention_strategy(name="perdim")
```

### Saving Training Outputs (Old vs New)

```python
# OLD WAY (in training script)
MODEL_DIR = Path("models/data")
np.save(MODEL_DIR / "book_embs.npy", embeddings)
with open(MODEL_DIR / "book_ids.json", "w") as f:
    json.dump(ids, f)

# NEW WAY
from models.core.paths import PATHS
PATHS.embeddings_dir.mkdir(parents=True, exist_ok=True)
np.save(PATHS.book_subject_embeddings, embeddings)
with open(PATHS.book_subject_ids, "w") as f:
    json.dump(ids, f)
```

---

## 🚀 Migration Checklist

### For Inference Code

- [ ] Replace hardcoded `"models/data/book_embs.npy"` with `PATHS.book_subject_embeddings`
- [ ] Replace hardcoded `"models/data/book_ids.json"` with `PATHS.book_subject_ids`
- [ ] Replace `"book_als_emb"` references with `"book_als_factors"`
- [ ] Replace `"user_als_emb"` references with `"user_als_factors"`
- [ ] Replace `"bayesian_tensor"` references with `"bayesian_scores"`
- [ ] Update attention path construction to use `PATHS.get_attention_path()`
- [ ] Update all `np.load()` and `json.load()` to use PATHS constants

### For Training Scripts

- [ ] Import `PATHS` from `models.core.paths`
- [ ] Update all save paths to use PATHS attributes
- [ ] Update all load paths to use PATHS attributes
- [ ] Update output print statements with new names
- [ ] Test script produces artifacts in correct location
- [ ] Verify ModelStore can load new artifacts

### For Tests

- [ ] Update fixture paths to use PATHS
- [ ] Update assertion checks for new filenames
- [ ] Add tests for path helper functions
- [ ] Verify backward compatibility if using facade

---

## 🎓 Understanding the Naming System

### Entity Types

| Entity | Meaning | Examples |
|--------|---------|----------|
| `book_*` | Book-level data | `book_subject_embeddings`, `book_als_factors` |
| `user_*` | User-level data | `user_als_factors`, `user_meta` |
| `subject_*` | Subject-level data | `subject_attention_*` |
| `bayesian_*` | Bayesian-derived | `bayesian_scores` |
| `gbt_*` | Gradient boosted tree model | `gbt_cold`, `gbt_warm` |

### Representation Types

| Type | Meaning | Examples |
|------|---------|----------|
| `*_embeddings` | Learned vector representations | `book_subject_embeddings` |
| `*_factors` | Matrix factorization latent factors | `book_als_factors` |
| `*_scores` | Precomputed ranking scores | `bayesian_scores` |
| `*_attention_*` | Attention mechanism components | `subject_attention_perdim` |
| `*_ids` | Index/ID mappings | `book_subject_ids`, `user_als_ids` |

### Variants

| Variant | Meaning | Examples |
|---------|---------|----------|
| `*_scalar` | Scalar attention pooling | `subject_attention_scalar` |
| `*_perdim` | Per-dimension attention | `subject_attention_perdim` |
| `*_selfattn` | Self-attention mechanism | `subject_attention_selfattn` |
| `*_cold` | Cold-start model | `gbt_cold` |
| `*_warm` | Warm-start model | `gbt_warm` |

---

## ❓ FAQ

### Q: Why not just use better folder names instead of longer filenames?

**A**: Folders organize by artifact *type* (embeddings, attention, scoring). Filenames describe *specific content* within that type. Both are needed for clarity.

### Q: Do I need to update my deployed models immediately?

**A**: No. The migration script creates both old and new versions. The ModelStore will try new paths first, then fall back to old paths during transition period.

### Q: What if I have custom scripts that reference old paths?

**A**: Use the `shared_utils.py` facade during migration. It re-exports everything from new locations. Update your scripts gradually, then remove the facade.

### Q: How do I know which name to use?

**A**: Ask: "What is this data?"
- Subject-pooled book vectors? → `book_subject_embeddings`
- ALS latent factors? → `book_als_factors`
- Precomputed scores? → `bayesian_scores`

### Q: Are these names too long?

**A**: They're verbose but *clear*. In code, you use path constants (`PATHS.book_subject_embeddings`) which are concise. The filename itself should be self-documenting.

---

## 📚 Related Documentation

- **Full Refactor Plan**: `models_refactor_plan_enhanced.md`
- **Migration Script**: `models/scripts/migrate_artifacts.py`
- **Path Definitions**: `models/core/paths.py`
- **Training Updates**: See Layer 6 in refactor plan

---

## ✅ Quick Wins

After this refactor:

1. **New team members** can understand artifacts without asking
2. **Code review** is easier (clear what's being loaded)
3. **Debugging** is faster (log messages are self-explanatory)
4. **Documentation** stays in sync (names match their purpose)
5. **Mistakes** are harder (wrong artifact name = immediate error)

---

*Last updated: Based on enhanced refactor plan*
