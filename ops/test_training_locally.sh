#!/bin/bash
# test_training_locally.sh
# Smoke test: checks that the training scripts run locally. Writes to
# models/artifacts/staging/ and does not promote. Requires uv sync --extra train.

set -e  # Exit on error

echo "=== Testing Training Pipeline Locally ==="
echo "PAD_IDX = ${PAD_IDX:-0}"

# Export data
echo "1. Exporting training data..."
uv run --extra train python -m models.training.export_training_data

# Test subject embeddings training
#echo "2. Testing subject embeddings..."
#uv run --extra train python models/training/train_subject_embs_contrastive.py --pad-idx ${PAD_IDX:-0}

# Precompute embeddings
echo "3. Precomputing book embeddings..."
uv run --extra train python models/training/precompute_embs.py --pad-idx ${PAD_IDX:-0}

# Precompute bayesian scores
echo "4. Precomputing Bayesian scores..."
uv run --extra train python models/training/precompute_bayesian.py --pad-idx ${PAD_IDX:-0}

# Train ALS
echo "5. Training ALS..."
uv run --extra train python models/training/train_als.py --pad-idx ${PAD_IDX:-0}

echo "=== All training scripts completed successfully! ==="
echo "Check models/artifacts/staging/ for outputs"
