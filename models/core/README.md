# models/core/

Core foundation layer providing constants, paths, and configuration for the models package.

## Overview

This module serves as the foundation for the entire models package. It provides:

- **Constants**: System-wide values like `PAD_IDX` and `DEVICE`
- **Paths**: Centralized registry of all model artifact locations
- **Config**: Environment variable handling and configuration management

## Module Structure

```
models/core/
├── __init__.py         # Public API exports
├── constants.py        # System constants (PAD_IDX, DEVICE)
├── paths.py            # Centralized path definitions
├── config.py           # Environment configuration
└── README.md           # This file
```

## Usage

### Constants

```python
from models.core import PAD_IDX, DEVICE

# Use in your code
padding_mask = (indices != PAD_IDX)
model = model.to(DEVICE)
```

### Paths

```python
from models.core import PATHS

# Load embeddings
import numpy as np
embeddings = np.load(PATHS.book_subject_embeddings)

# Get attention model path by strategy
attention_path = PATHS.get_attention_path("perdim")

# Ensure directories exist before saving
PATHS.ensure_artifact_dirs()
np.save(PATHS.book_als_factors, factors)
```

### Configuration

```python
from models.core import Config

# Get current attention strategy
strategy = Config.get_attention_strategy()  # from ATTN_STRATEGY env var

# Get training configuration
train_config = Config.get_training_config()
lr = train_config["subject_learning_rate"]

# Get contrastive learning config
contrast_config = Config.get_contrastive_config()
temperature = contrast_config["contrast_temperature"]
```

## Path Registry

All model artifacts follow the naming pattern: `{entity}_{representation_type}_{optional_variant}`

### Embeddings Directory

| Artifact | Path Property | Description |
|----------|---------------|-------------|
| `book_subject_embeddings.npy` | `PATHS.book_subject_embeddings` | Attention-pooled book embeddings from subjects |
| `book_subject_ids.json` | `PATHS.book_subject_ids` | Item indices for book embeddings |
| `book_als_factors.npy` | `PATHS.book_als_factors` | ALS latent factors for books |
| `book_als_ids.json` | `PATHS.book_als_ids` | Item indices for ALS factors |
| `user_als_factors.npy` | `PATHS.user_als_factors` | ALS latent factors for users |
| `user_als_ids.json` | `PATHS.user_als_ids` | User IDs for ALS factors |

### Attention Directory

| Artifact | Path Property | Description |
|----------|---------------|-------------|
| `subject_attention_scalar.pth` | `PATHS.subject_attention_scalar` | Scalar attention pooling |
| `subject_attention_perdim.pth` | `PATHS.subject_attention_perdim` | Per-dimension attention |
| `subject_attention_selfattn.pth` | `PATHS.subject_attention_selfattn` | Self-attention pooling |
| `subject_attention_selfattn_perdim.pth` | `PATHS.subject_attention_selfattn_perdim` | Self-attention + per-dim |

Use `PATHS.get_attention_path(strategy)` for dynamic strategy selection.

### Scoring Directory

| Artifact | Path Property | Description |
|----------|---------------|-------------|
| `bayesian_scores.npy` | `PATHS.bayesian_scores` | Precomputed Bayesian popularity scores |
| `gbt_cold.pickle` | `PATHS.gbt_cold` | GBT model for cold-start users |
| `gbt_warm.pickle` | `PATHS.gbt_warm` | GBT model for warm-start users |

### Training Data Directory

Access training data via:
- `PATHS.training_interactions`
- `PATHS.training_users`
- `PATHS.training_books`
- `PATHS.training_book_subjects`
- `PATHS.training_user_fav_subjects`
- `PATHS.training_subjects`

## Design Principles

1. **Single Source of Truth**: All paths defined in one place
2. **Type Safety**: Proper type hints for path methods
3. **Validation**: Runtime validation for dynamic paths (e.g., attention strategies)
4. **Convenience**: Both direct properties and dynamic getters available
5. **Discoverability**: IDE autocomplete shows all available paths

## Environment Variables

Handled by `Config` class:

| Variable | Default | Description |
|----------|---------|-------------|
| `PAD_IDX` | `0` | Padding index for embeddings |
| `ATTN_STRATEGY` | `"scalar"` | Attention pooling strategy |
| `SUBJ_EMB_DIM` | `64` | Subject embedding dimension |
| `SUBJ_DROPOUT` | `0.3` | Dropout rate for attention |
| `SUBJ_N_HEADS` | `4` | Number of attention heads |
| `SUBJ_BS` | `1024` | Training batch size |
| `SUBJ_LR` | `3e-3` | Learning rate |
| `SUBJ_EPOCHS` | `14` | Number of training epochs |
| `SUBJECT_AUTO_TRAIN` | `"false"` | Enable automatic training |
| `LAMBDA_CONTRAST` | `0.8` | Contrastive loss weight |
| `LAMBDA_MSE` | `0.2` | MSE loss weight |
| `CONTRAST_T` | `0.07` | Contrastive temperature |
| `CONTRAST_USE_JACCARD` | `"1"` | Use Jaccard similarity |
| `CONTRAST_OVERLAP_THRESH` | `2` | Overlap threshold |

## Migration from Old Code

Old hardcoded paths should be replaced with `PATHS` properties:

```python
# OLD
embeddings = np.load("models/data/book_embs.npy")
with open("models/data/book_ids.json") as f:
    ids = json.load(f)

# NEW
embeddings = np.load(PATHS.book_subject_embeddings)
with open(PATHS.book_subject_ids) as f:
    ids = json.load(f)
```

## Related Documentation

- See `models/data/README.md` for data loading utilities
- See `models_refactor_plan.md` for overall refactor strategy
- See `naming_references.md` for complete artifact naming guide
