# ML Models Module Refactor Plan (Enhanced)

## Refactor Strategy: Bottom-Up by Component
**Principle**: Touch each file once, do it completely. Work from foundation to API in dependency order.

---

## Naming Conventions Standardization

### Current Problems
- `book_embs.npy` - Actually subject-pooled embeddings, misleading name
- Multiple "book embedding" types without clear distinction
- Inconsistent naming patterns across artifact types

### New Naming Scheme

#### General Patterns
```
{entity}_{representation_type}_{optional_variant}.{ext}

Examples:
- book_subject_embeddings.npy    (entity=book, type=subject embeddings)
- book_als_factors.npy           (entity=book, type=als factors)
- bayesian_scores.npy            (purpose-based name for precomputed scores)
```

#### Complete Artifact Renaming Map

**Embeddings Directory**
| Old Name | New Name | Reason |
|----------|----------|--------|
| `book_embs.npy` | `book_subject_embeddings.npy` | Clarifies these are subject-pooled |
| `book_ids.json` | `book_subject_ids.json` | Matches embedding file |
| `book_als_emb.npy` | `book_als_factors.npy` | "factors" is more accurate for ALS |
| `book_als_ids.json` | `book_als_ids.json` | ✓ Keep (already clear) |
| `user_als_emb.npy` | `user_als_factors.npy` | Consistent with book ALS |
| `user_als_ids.json` | `user_als_ids.json` | ✓ Keep (already clear) |

**Attention Components Directory**
| Old Name | New Name | Reason |
|----------|----------|--------|
| `subject_attention_components.pth` | `subject_attention_scalar.pth` | Shorter, clearer variant |
| `subject_attention_components_perdim.pth` | `subject_attention_perdim.pth` | Remove redundant "components" |
| `subject_attention_components_selfattn.pth` | `subject_attention_selfattn.pth` | Remove redundant "components" |
| `subject_attention_components_selfattn_perdim.pth` | `subject_attention_selfattn_perdim.pth` | Remove redundant "components" |

**Scoring Directory**
| Old Name | New Name | Reason |
|----------|----------|--------|
| `bayesian_tensor.npy` | `bayesian_scores.npy` | More descriptive of contents |
| `gbt_cold.pickle` | `gbt_cold.pickle` | ✓ Keep (already clear) |
| `gbt_warm.pickle` | `gbt_warm.pickle` | ✓ Keep (already clear) |

---

## Component Order & Dependencies

```
Layer 5: API Routes
         ↓ depends on
Layer 4: Engines (recommendation_engine, recommender_strategy)
         ↓ depends on
Layer 3: Strategies (similarity, candidates, rerankers)
         ↓ depends on
Layer 2: Core Logic (embeddings, queries, subject_utils)
         ↓ depends on
Layer 1: Data Loading (ModelStore, loaders)
         ↓ depends on
Layer 0: Foundation (constants, paths, config)
```

**Work order**: 0 → 1 → 2 → 3 → 4 → 5 → 6 (Training)

---

## Layer 0: Foundation (No dependencies)

### Files Created
```
models/
└── core/
    ├── __init__.py
    ├── constants.py       # PAD_IDX
    ├── paths.py          # All artifact paths centralized with NEW NAMES
    └── config.py         # Optional: pydantic config class
```

### paths.py Structure
```python
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).parent.parent.parent
MODELS_ROOT = REPO_ROOT / "models"

@dataclass(frozen=True)
class ArtifactPaths:
    """Centralized path definitions for all model artifacts."""

    # Root directories
    artifacts_root: Path = MODELS_ROOT / "artifacts"
    embeddings_dir: Path = artifacts_root / "embeddings"
    attention_dir: Path = artifacts_root / "attention"
    scoring_dir: Path = artifacts_root / "scoring"

    # Subject-based book embeddings (pooled from subjects)
    book_subject_embeddings: Path = embeddings_dir / "book_subject_embeddings.npy"
    book_subject_ids: Path = embeddings_dir / "book_subject_ids.json"

    # ALS factorization embeddings
    book_als_factors: Path = embeddings_dir / "book_als_factors.npy"
    book_als_ids: Path = embeddings_dir / "book_als_ids.json"
    user_als_factors: Path = embeddings_dir / "user_als_factors.npy"
    user_als_ids: Path = embeddings_dir / "user_als_ids.json"

    # Attention pooling components (by strategy)
    def get_attention_path(self, strategy: str = "scalar") -> Path:
        """Get path for attention components by strategy name."""
        return self.attention_dir / f"subject_attention_{strategy}.pth"

    # Precomputed scores
    bayesian_scores: Path = scoring_dir / "bayesian_scores.npy"
    gbt_cold_model: Path = scoring_dir / "gbt_cold.pickle"
    gbt_warm_model: Path = scoring_dir / "gbt_warm.pickle"

    # Training data (unchanged location)
    training_data_dir: Path = MODELS_ROOT / "training" / "data"

PATHS = ArtifactPaths()
```

### Physical Artifact Reorganization
```
models/artifacts/          # NEW directory
├── embeddings/
│   ├── book_subject_embeddings.npy      # RENAMED from book_embs.npy
│   ├── book_subject_ids.json            # RENAMED from book_ids.json
│   ├── book_als_factors.npy             # RENAMED from book_als_emb.npy
│   ├── book_als_ids.json                # (kept same)
│   ├── user_als_factors.npy             # RENAMED from user_als_emb.npy
│   └── user_als_ids.json                # (kept same)
├── attention/
│   ├── subject_attention_scalar.pth     # RENAMED (removed _components)
│   ├── subject_attention_perdim.pth     # RENAMED
│   ├── subject_attention_selfattn.pth   # RENAMED
│   └── subject_attention_selfattn_perdim.pth  # RENAMED
└── scoring/
    ├── bayesian_scores.npy              # RENAMED from bayesian_tensor.npy
    ├── gbt_cold.pickle                  # (kept same)
    └── gbt_warm.pickle                  # (kept same)
```

### Migration Script for Layer 0
```python
# models/scripts/migrate_artifacts.py
"""One-time script to rename and reorganize artifacts."""

import shutil
from pathlib import Path
from models.core.paths import PATHS, MODELS_ROOT

OLD_DATA_DIR = MODELS_ROOT / "data"

RENAME_MAP = {
    # Embeddings
    OLD_DATA_DIR / "book_embs.npy": PATHS.book_subject_embeddings,
    OLD_DATA_DIR / "book_ids.json": PATHS.book_subject_ids,
    OLD_DATA_DIR / "book_als_emb.npy": PATHS.book_als_factors,
    OLD_DATA_DIR / "user_als_emb.npy": PATHS.user_als_factors,

    # Attention (just move, names change)
    OLD_DATA_DIR / "subject_attention_components.pth":
        PATHS.get_attention_path("scalar"),
    OLD_DATA_DIR / "subject_attention_components_perdim.pth":
        PATHS.get_attention_path("perdim"),
    OLD_DATA_DIR / "subject_attention_components_selfattn.pth":
        PATHS.get_attention_path("selfattn"),
    OLD_DATA_DIR / "subject_attention_components_selfattn_perdim.pth":
        PATHS.get_attention_path("selfattn_perdim"),

    # Scoring
    OLD_DATA_DIR / "bayesian_tensor.npy": PATHS.bayesian_scores,
    OLD_DATA_DIR / "gbt_cold.pickle": PATHS.gbt_cold_model,
    OLD_DATA_DIR / "gbt_warm.pickle": PATHS.gbt_warm_model,

    # IDs that don't change name but move location
    OLD_DATA_DIR / "book_als_ids.json": PATHS.book_als_ids,
    OLD_DATA_DIR / "user_als_ids.json": PATHS.user_als_ids,
}

def migrate():
    """Move and rename artifacts."""
    # Create new directories
    PATHS.embeddings_dir.mkdir(parents=True, exist_ok=True)
    PATHS.attention_dir.mkdir(parents=True, exist_ok=True)
    PATHS.scoring_dir.mkdir(parents=True, exist_ok=True)

    for old_path, new_path in RENAME_MAP.items():
        if old_path.exists():
            print(f"Moving: {old_path.name} -> {new_path}")
            shutil.copy2(old_path, new_path)
        else:
            print(f"Warning: {old_path} not found")

    print("\n✅ Migration complete!")
    print(f"New artifacts location: {PATHS.artifacts_root}")
    print("\nOld files still in place. Remove manually after verification.")

if __name__ == "__main__":
    migrate()
```

### Deliverable
- ✅ New foundation files exist
- ✅ Artifacts physically moved to new structure
- ✅ All names clarified and conventional
- ✅ Old files kept for rollback
- ✅ No other files changed yet

---

## Layer 1: Data Loading (Depends on: Layer 0)

### Files Created/Modified
```
models/
├── data/
│   ├── __init__.py
│   ├── loaders.py        # NEW: ModelStore + load functions with NEW PATHS
│   └── queries.py        # NEW: DB/DataFrame query operations
└── shared_utils.py       # MODIFIED: Facade for backward compatibility
```

### Key Changes to loaders.py

```python
# models/data/loaders.py
"""Data loading utilities with updated paths."""

from models.core.paths import PATHS
from models.core.constants import PAD_IDX
import numpy as np
import json

def load_book_subject_embeddings():
    """Load subject-pooled book embeddings (formerly book_embs)."""
    embs = np.load(PATHS.book_subject_embeddings)
    with open(PATHS.book_subject_ids) as f:
        book_ids = json.load(f)
    return embs, book_ids

def load_als_embeddings():
    """Load ALS factorization matrices."""
    user_factors = np.load(PATHS.user_als_factors)
    book_factors = np.load(PATHS.book_als_factors)

    with open(PATHS.user_als_ids) as f:
        user_ids = json.load(f)
    with open(PATHS.book_als_ids) as f:
        book_ids = json.load(f)

    return user_factors, book_factors, user_ids, book_ids

class ModelStore:
    """Singleton for lazy-loading all model artifacts."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        # Core caches - UPDATED NAMING
        self._book_subject_embs = None       # was _book_embs
        self._book_subject_embs_norm = None  # was _book_embs_norm
        self._book_subject_ids = None        # was _book_ids

        # ALS - UPDATED NAMING
        self._user_als_factors = None        # was _user_als_embs
        self._book_als_factors = None        # was _book_als_embs
        self._book_als_factors_norm = None   # was _book_als_embs_norm

        # Other caches (names unchanged)
        self._bayesian_scores = None         # was _bayesian_tensor
        self._book_meta = None
        self._user_meta = None
        self._book_to_subj = None
        self._item_idx_to_row = None

        # Lookup maps
        self._user_id_to_als_row = None
        self._book_row_to_item_idx = None
        self._book_als_ids = None
        self._als_book_id_set = None

        # Attention strategy cache
        self._attn_strategy = None
        self._attn_strategy_name = None

    def get_book_subject_embeddings(self, normalized: bool = False):
        """Get subject-pooled book embeddings."""
        if self._book_subject_embs is None:
            self._book_subject_embs, self._book_subject_ids = load_book_subject_embeddings()
            self._item_idx_to_row = {idx: i for i, idx in enumerate(self._book_subject_ids)}

        if not normalized:
            return self._book_subject_embs, self._book_subject_ids

        if self._book_subject_embs_norm is None:
            embs = self._book_subject_embs
            norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            self._book_subject_embs_norm = norm.astype(np.float32, copy=False)

        return self._book_subject_embs_norm, self._book_subject_ids

    def get_als_factors(self, normalized: bool = False):
        """Get ALS factorization matrices."""
        if self._user_als_factors is None:
            (self._user_als_factors, self._book_als_factors,
             user_ids, self._book_als_ids) = load_als_embeddings()

            self._user_id_to_als_row = {uid: i for i, uid in enumerate(user_ids)}
            self._book_row_to_item_idx = {i: iid for i, iid in enumerate(self._book_als_ids)}

        if not normalized:
            return (self._user_als_factors, self._book_als_factors,
                    self._user_id_to_als_row, self._book_row_to_item_idx)

        if self._book_als_factors_norm is None:
            embs = self._book_als_factors
            norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            self._book_als_factors_norm = norm.astype(np.float32, copy=False)

        return (self._user_als_factors, self._book_als_factors_norm,
                self._user_id_to_als_row, self._book_row_to_item_idx)

    def get_bayesian_scores(self):
        """Load precomputed Bayesian popularity scores."""
        if self._bayesian_scores is None:
            arr = np.load(PATHS.bayesian_scores)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            self._bayesian_scores = arr
        return self._bayesian_scores

    def get_attention_strategy(self, name=None):
        """Load attention pooling strategy by name."""
        name = name or os.getenv("ATTN_STRATEGY", "scalar")

        if self._attn_strategy_name != name:
            from models.subject_attention_strategy import STRATEGY_REGISTRY

            if name not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown attention strategy: {name}")

            strategy_class = STRATEGY_REGISTRY[name]
            path = PATHS.get_attention_path(name)

            self._attn_strategy = strategy_class(path=str(path))
            self._attn_strategy_name = name

        return self._attn_strategy

    # ... other methods stay same, just update internal variable names ...

    @classmethod
    def reset(cls):
        """Hard reset singleton."""
        cls._instance = None
```

### Backward Compatibility Facade

```python
# models/shared_utils.py (MODIFIED)
"""
DEPRECATED: Backward compatibility facade.
Import from models.data.loaders or models.data.queries instead.
This file will be removed in a future version.
"""

import warnings

# Re-export everything from new locations
from models.data.loaders import (
    ModelStore,
    load_book_subject_embeddings as load_book_embeddings,  # alias
)
from models.data.queries import (
    get_read_books,
    get_candidate_book_df,
    filter_read_books,
    add_book_embeddings,
    get_user_num_ratings,
)
from models.core.embeddings import get_user_embedding, normalize_vector
from models.core.subject_utils import compute_subject_overlap
from models.utils.tensor_ops import decompose_embeddings, clean_row
from models.core.constants import PAD_IDX

warnings.warn(
    "shared_utils.py is deprecated. Import from specific modules instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### Deliverable
- ✅ ModelStore works with new paths and names
- ✅ All loading functions updated
- ✅ Backward compatibility maintained via facade
- ✅ Clear deprecation path

---

## Layer 2-5: (Same as original plan, just update imports)

_Content identical to original plan but with updated import paths and names_

---

## Layer 6: Training Scripts Update (NEW)

### Training Scripts to Update

All scripts in `models/training/` that generate artifacts:

1. `precompute_embs.py` - Generates subject embeddings
2. `precompute_bayesian.py` - Generates Bayesian scores
3. `train_als.py` - Generates ALS factors
4. `train_subject_attention.py` - Base pooler components
5. `train_subjects_embs.py` - Supervised attention training
6. `train_subject_embs_contrastive.py` - Contrastive attention training
7. `train_cold_gbt.py` - Cold start GBT model (uses embeddings)
8. `train_warm_gbt.py` - Warm GBT model (uses embeddings)

### Update Pattern for Training Scripts

```python
# OLD WAY (every training script)
import os
REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = REPO_ROOT / "models/data"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save outputs
np.save(MODEL_DIR / "book_embs.npy", embeddings)
with open(MODEL_DIR / "book_ids.json", "w") as f:
    json.dump(book_ids, f)

# NEW WAY
from models.core.paths import PATHS

# Directories auto-created by PATHS
PATHS.embeddings_dir.mkdir(parents=True, exist_ok=True)

# Save with new names
np.save(PATHS.book_subject_embeddings, embeddings)
with open(PATHS.book_subject_ids, "w") as f:
    json.dump(book_ids, f)
```

### Specific Script Updates

#### precompute_embs.py
```python
# CHANGES:
# - Import PATHS
# - Save to book_subject_embeddings.npy (not book_embs.npy)
# - Save to book_subject_ids.json (not book_ids.json)

from models.core.paths import PATHS

# ... existing computation code ...

# NEW save section:
PATHS.embeddings_dir.mkdir(parents=True, exist_ok=True)
np.save(PATHS.book_subject_embeddings, pooled_embs)
with open(PATHS.book_subject_ids, "w") as f:
    json.dump(book_ids, f)

print("✅ Saved:")
print(f"   - {PATHS.book_subject_embeddings}")
print(f"   - {PATHS.book_subject_ids}")
```

#### precompute_bayesian.py
```python
# CHANGES:
# - Import PATHS
# - Load from book_subject_ids.json (not book_ids.json)
# - Save to bayesian_scores.npy (not bayesian_tensor.npy)

from models.core.paths import PATHS

# Load book IDs
with open(PATHS.book_subject_ids) as f:
    book_ids = json.load(f)

# ... compute scores ...

# Save
PATHS.scoring_dir.mkdir(parents=True, exist_ok=True)
np.save(PATHS.bayesian_scores, bayesian_tensor)
print(f"✅ Saved: {PATHS.bayesian_scores}")
```

#### train_als.py
```python
# CHANGES:
# - Import PATHS
# - Save to *_als_factors.npy (not *_als_emb.npy)

from models.core.paths import PATHS

# ... train model ...

print("💾 Saving outputs...")
PATHS.embeddings_dir.mkdir(parents=True, exist_ok=True)

np.save(PATHS.user_als_factors, model.user_factors)
np.save(PATHS.book_als_factors, model.item_factors)

with open(PATHS.user_als_ids, "w") as f:
    json.dump([int(idx2user[i]) for i in range(num_users)], f)
with open(PATHS.book_als_ids, "w") as f:
    json.dump([int(idx2item[i]) for i in range(num_items)], f)

print("✅ ALS training complete.")
```

#### train_subject_attention.py (and variants)
```python
# CHANGES:
# - Import PATHS
# - Use get_attention_path() for strategy-specific paths
# - Remove _components from filename

from models.core.paths import PATHS

# Filename map for output
OUT_NAME_BY_KIND = {
    "scalar": "scalar",           # CHANGED: was "subject_attention_components.pth"
    "perdim": "perdim",           # CHANGED: simplified
    "selfattn": "selfattn",       # CHANGED
    "selfattn_perdim": "selfattn_perdim",  # CHANGED
}

def save_components(pooler, kind):
    """Save attention components with new naming."""
    state = {}  # ... build state dict ...

    # NEW: Use PATHS helper
    PATHS.attention_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.get_attention_path(kind)

    torch.save(state, out_path)
    print(f"✅ Saved to {out_path}")
```

#### train_cold_gbt.py & train_warm_gbt.py
```python
# CHANGES:
# - Import PATHS
# - Load embeddings using new names
# - Save models to PATHS.gbt_*_model

from models.core.paths import PATHS
from models.data.loaders import load_book_subject_embeddings

# Load embeddings
book_embs, book_ids = load_book_subject_embeddings()

# ... training ...

# Save
PATHS.scoring_dir.mkdir(parents=True, exist_ok=True)
with open(PATHS.gbt_cold_model, "wb") as f:  # or gbt_warm_model
    pickle.dump(model, f)

print(f"✅ Model saved to: {PATHS.gbt_cold_model}")
```

### Updated Automated Training Script

```python
# ops/automated_training.py
"""
UPDATED: Now handles new artifact structure and paths
"""

import subprocess
import json
import os, sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import shlex

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import new paths system
from models.core.paths import PATHS

load_dotenv()

# Azure & Remote config (unchanged)
AZURE_AUTH = os.getenv("AZURE_AUTH_LOCATION")
AZURE_VM = os.getenv("AZURE_VM_NAME")
AZURE_RG = os.getenv("AZURE_RESOURCE_GROUP")
REMOTE_HOST = os.getenv("REMOTE_HOST")
REMOTE_REPO = os.getenv("REMOTE_REPO_PATH")
REMOTE_BACKUP_DIR = os.getenv('REMOTE_BACKUP_DIR')

with open(AZURE_AUTH) as f:
    sp = json.load(f)

AZURE_CLIENT_ID = sp["clientId"]
AZURE_CLIENT_SECRET = sp["clientSecret"]
AZURE_TENANT_ID = sp["tenantId"]

# Local/Remote paths - UPDATED
TRAINING_DATA_NEW = PATHS.training_data_dir / "new_data"
TRAINING_DATA_MAIN = PATHS.training_data_dir
REMOTE_DATA = f"{REMOTE_HOST}:{REMOTE_REPO}/models/training/data"

# NEW: Use artifacts directory instead of old data directory
REMOTE_ARTIFACTS = f"{REMOTE_HOST}:{REMOTE_REPO}/models/artifacts"
LOCAL_ARTIFACTS = PATHS.artifacts_root

LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR", PATHS.training_data_dir.parent / "logs"))

# Training scripts - UPDATED
ATTN_STRATEGY = os.getenv("ATTN_STRATEGY", "scalar").lower()
train_subject_script = os.getenv("SUBJECT_TRAIN_FILE", 'train_subjects_embs.py')
SUBJECT_AUTO_TRAIN = os.getenv("SUBJECT_AUTO_TRAIN", "false")

TRAIN_SCRIPTS = []
if SUBJECT_AUTO_TRAIN.lower() == "true":
    TRAIN_SCRIPTS.append(train_subject_script)

TRAIN_SCRIPTS.extend([
    "precompute_embs.py",
    "precompute_bayesian.py",
    "train_als.py",
    # Add GBT training if needed
    # "train_cold_gbt.py",
    # "train_warm_gbt.py",
])

def run(cmd, **kwargs):
    """Execute shell command with logging."""
    print(f"▶ Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)

def sync_artifacts_to_remote():
    """Copy newly trained artifacts to training server."""
    print("📤 Syncing training data to remote...")
    run(f"scp -r {TRAINING_DATA_NEW}/* {REMOTE_DATA}/")

def sync_artifacts_from_remote():
    """Copy trained artifacts back from training server."""
    print("📥 Copying trained artifacts back...")

    # NEW: Create local artifacts directory structure
    LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Sync entire artifacts directory
    run(f"scp -r {REMOTE_ARTIFACTS}/* {LOCAL_ARTIFACTS}/")

    print(f"✅ Artifacts synced to: {LOCAL_ARTIFACTS}")

def replace_training_data():
    """Move new training data to main location."""
    print("📝 Replacing old training data with new data...")

    # Remove old pickle files
    for file in TRAINING_DATA_MAIN.glob("*.pkl"):
        if file.is_file():
            file.unlink()

    # Move new files
    for file in TRAINING_DATA_NEW.glob("*.pkl"):
        shutil.move(str(file), TRAINING_DATA_MAIN)

    print("✅ Training data updated")

def reload_api_models():
    """Trigger API to reload models from new artifacts."""
    print("🧠 Reloading models in API memory...")

    api_url = os.getenv("RELOAD_API_URL", "http://localhost:8000/admin/reload_models")
    admin_secret = os.getenv("ADMIN_SECRET")

    try:
        resp = requests.post(api_url, params={"secret": admin_secret})
        if resp.status_code == 200:
            print("✅ API reload successful.")
        else:
            print(f"❌ API reload failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"❌ Exception during reload: {e}")

def update_meilisearch():
    """Update Bayesian scores in search index."""
    print("📊 Updating bayes_pop in Meilisearch...")
    run(f"python {PROJECT_ROOT}/ops/meilisearch/update_bayes_pop.py")

def main():
    """Main automated training pipeline."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_file_local = LOG_DIR / f"{timestamp}_train.log"
    log_file_remote = f"{REMOTE_REPO}/models/training/logs/{timestamp}_train.log"

    # 1. Azure authentication
    print("📡 Authenticating with Azure...")
    run(f'az login --service-principal -u {AZURE_CLIENT_ID} -p {AZURE_CLIENT_SECRET} --tenant {AZURE_TENANT_ID}')

    # 2. Start training VM
    print("🚀 Starting training VM...")
    run(f'az vm start --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    # 3. Export new training data
    print("📦 Exporting training data locally...")
    run("python models/training/export_training_data.py")

    # 4. Wait for SSH
    print("⌛ Waiting for SSH to be available...")
    while subprocess.run(f"ssh -o BatchMode=no {REMOTE_HOST} 'echo ready'",
                        shell=True).returncode != 0:
        print("  - Still waiting for SSH...")
        time.sleep(5)

    # 5. Backup current database
    print("📄 Backing up current database...")
    run(f'python {PROJECT_ROOT}/ops/backup_db.py')

    # Copy backup to remote if exists
    def _db_from_env():
        db = os.getenv("MYSQL_DB", "").strip()
        if db:
            return db
        try:
            from urllib.parse import urlsplit
            return (urlsplit(os.getenv("DATABASE_URL","")).path or "").lstrip("/")
        except Exception:
            return ""

    db_name = _db_from_env()
    backup_dir = PROJECT_ROOT / "data/backups/db"
    candidates = sorted(backup_dir.glob(f"*_{db_name}.sql.gz"))
    if candidates:
        latest_dump = candidates[-1]
        run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_BACKUP_DIR}'")
        run(f"scp {shlex.quote(str(latest_dump))} {REMOTE_HOST}:{REMOTE_BACKUP_DIR.rstrip('/')}/")

    # 6. Sync training data to remote
    sync_artifacts_to_remote()

    # 7. Run training scripts remotely
    print("⚙️ Running training scripts remotely...")
    for script in TRAIN_SCRIPTS:
        print(f"➡ {script}")
        cmd = (
            f"ssh {REMOTE_HOST} "
            f"'cd {REMOTE_REPO} && "
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate bookrec-api && "
            f"python models/training/{script}'"
        )

        # Stream output and log
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True) as proc:
            lines = []
            for line in proc.stdout:
                print(line, end="")
                lines.append(line)

            # Save logs remotely
            run(f"ssh {REMOTE_HOST} 'mkdir -p {REMOTE_REPO}/models/training/logs'")
            joined = "".join(lines).replace("'", "'\\''")
            run(f"ssh {REMOTE_HOST} \"echo '{joined}' >> {log_file_remote}\"")

        # Save logs locally
        with open(log_file_local, "a") as f:
            f.writelines(lines)

    # 8. Copy trained artifacts back
    sync_artifacts_from_remote()

    # 9. Stop training VM
    print("🛑 Stopping training VM...")
    run(f'az vm deallocate --name {AZURE_VM} --resource-group "{AZURE_RG}"')

    # 10. Replace training data
    replace_training_data()

    # 11. Reload API models
    reload_api_models()

    # 12. Update search index
    update_meilisearch()

    print("✅ Done. Logs saved to:", log_file_local)
    print(f"✅ New artifacts location: {LOCAL_ARTIFACTS}")

if __name__ == "__main__":
    main()
```

### Migration Checklist for Training Scripts

```markdown
## Training Scripts Migration Checklist

### ✅ Phase 1: Update Imports
- [ ] precompute_embs.py
- [ ] precompute_bayesian.py
- [ ] train_als.py
- [ ] train_subject_attention.py
- [ ] train_subjects_embs.py
- [ ] train_subject_embs_contrastive.py
- [ ] train_cold_gbt.py
- [ ] train_warm_gbt.py

### ✅ Phase 2: Update Output Paths
- [ ] All scripts use PATHS.embeddings_dir
- [ ] All scripts use PATHS.attention_dir
- [ ] All scripts use PATHS.scoring_dir
- [ ] All scripts use specific named paths (no hardcoded strings)

### ✅ Phase 3: Update Input Paths
- [ ] Scripts loading embeddings use new names
- [ ] Scripts loading attention use get_attention_path()
- [ ] Scripts loading Bayesian use bayesian_scores

### ✅ Phase 4: Test Locally
- [ ] Run each script in isolation
- [ ] Verify outputs have correct names
- [ ] Verify outputs in correct directories
- [ ] Check ModelStore can load new artifacts

### ✅ Phase 5: Update Automated Training
- [ ] Update automated_training.py paths
- [ ] Update sync functions
- [ ] Update remote artifact paths
- [ ] Test full pipeline on VM

### ✅ Phase 6: Documentation
- [ ] Update training README
- [ ] Document new artifact structure
- [ ] Document migration process
- [ ] Update deployment docs
```

---

## Complete Migration Schedule (Updated)

### Day 1: Foundation & Migration
- ✅ Create Layer 0 files (paths, constants, config)
- ✅ Run migration script to move artifacts
- ✅ Verify new structure
- ✅ Test imports

### Day 2: Data Loading
- ✅ Create Layer 1 files (loaders.py, queries.py)
- ✅ Update ModelStore with new paths/names
- ✅ Create shared_utils facade
- ✅ Test loading from new locations

### Day 3: Core Logic
- ✅ Create Layer 2 files (embeddings.py, subject_utils.py, tensor_ops.py)
- ✅ Move functions from shared_utils
- ✅ Update facade
- ✅ Test functions

### Day 4: Strategies
- ✅ Update Layer 3 files (similarity, candidates, rerankers)
- ✅ Remove singletons
- ✅ Move embedding computation
- ✅ Test strategies

### Day 5: Engines & API
- ✅ Update Layer 4 files (engines)
- ✅ Update Layer 5 files (API routes)
- ✅ Test end-to-end

### Day 6: Training Scripts (Phase 1)
- ✅ Update precompute_embs.py
- ✅ Update precompute_bayesian.py
- ✅ Update train_als.py
- ✅ Test each script locally
- ✅ Verify artifacts generated correctly

### Day 7: Training Scripts (Phase 2)
- ✅ Update attention training scripts
- ✅ Update GBT training scripts
- ✅ Test all training locally
- ✅ Verify inference still works

### Day 8: Automated Training
- ✅ Update automated_training.py
- ✅ Test on training VM
- ✅ Verify artifact sync
- ✅ Full end-to-end test

### Day 9: Cleanup & Validation
- ✅ Remove shared_utils facade
- ✅ Update all remaining imports
- ✅ Full regression test
- ✅ Performance benchmarks
- ✅ Documentation update

---

## Success Criteria (Updated)

### Code Quality
- ✅ No file over 300 lines
- ✅ Zero circular imports
- ✅ Clear module boundaries
- ✅ Easy to find any logic

### Performance
- ✅ Warm user latency improved 15-30%
- ✅ Memory usage reduced ~200MB
- ✅ Cold start time unchanged

### Correctness
- ✅ All existing tests pass
- ✅ Training produces identical artifacts (modulo randomness)
- ✅ Inference produces identical results
- ✅ Automated training pipeline works end-to-end

### Naming Clarity
- ✅ All artifact names clearly indicate content
- ✅ No ambiguous "book_embs" references
- ✅ Subject embeddings clearly distinguished from ALS
- ✅ Conventional naming patterns throughout

---

## Rollback Plan

1. **Git tags**: Tag before each layer
2. **Artifact backup**: Keep old `models/data/` dir until Day 9
3. **Dual loading**: ModelStore tries new paths, falls back to old
4. **Training rollback**: Keep old scripts in `training/legacy/` dir
5. **Quick revert**: Script to move artifacts back to old structure

```python
# rollback.py
def rollback_artifacts():
    """Emergency rollback to old structure."""
    import shutil
    from models.core.paths import PATHS, MODELS_ROOT

    OLD_DATA_DIR = MODELS_ROOT / "data"
    OLD_DATA_DIR.mkdir(exist_ok=True)

    # Copy everything back
    for file in PATHS.artifacts_root.rglob("*"):
        if file.is_file():
            # Reverse the rename map
            old_name = get_old_name(file.name)  # Look up original name
            shutil.copy2(file, OLD_DATA_DIR / old_name)

    print("✅ Rolled back to old structure")
```

---

## Open Questions

1. **Timing**: Should we rename artifacts in one PR or gradually?
   - **Recommendation**: One PR with dual loading for safety

2. **Semantic Index**: Should we also reorganize semantic index directories?
   - **Current**: `models/data/enriched_v2/semantic.faiss`
   - **Proposed**: Keep separate (different system)

3. **Test Coverage**: Add tests now or after refactor?
   - **Recommendation**: Add critical path tests during Day 6-7

4. **Documentation**: Update docs during or after?
   - **Recommendation**: Update during (prevents drift)

5. **Deprecation Period**: How long to keep facade?
   - **Recommendation**: 2 sprints, then remove

---

## Summary

This enhanced plan adds:

1. **Clear naming conventions** for all artifacts
2. **Complete training script updates** with new paths
3. **Automated training pipeline** updates
4. **Migration scripts** for artifact reorganization
5. **Extended schedule** to include training (9 days total)
6. **Additional success criteria** around naming clarity
7. **Rollback procedures** for training artifacts

The key insight: **Training scripts are first-class citizens** in this refactor. They must be updated atomically with the inference code to maintain consistency.
