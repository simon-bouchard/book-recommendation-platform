#!/bin/bash
# test_training_locally.sh
# Quick test of training pipeline locally

set -e  # Exit on error

echo "=== Testing Training Pipeline Locally ==="
echo "PAD_IDX = ${PAD_IDX:-0}"

# Export data
echo "1. Exporting training data..."
python models/training/export_training_data.py

# Test subject embeddings training
#echo "2. Testing subject embeddings..."
#python models/training/train_subject_embs_contrastive.py --pad-idx ${PAD_IDX:-0}

# Precompute embeddings
echo "3. Precomputing book embeddings..."
python models/training/precompute_embs.py --pad-idx ${PAD_IDX:-0}

# Precompute bayesian scores
echo "4. Precomputing Bayesian scores..."
python models/training/precompute_bayesian.py --pad-idx ${PAD_IDX:-0}

# Train ALS
echo "5. Training ALS..."
python models/training/train_als.py --pad-idx ${PAD_IDX:-0}

echo "=== All training scripts completed successfully! ==="
echo "Check models/data/ for outputs"
