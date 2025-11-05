#!/bin/bash

# Set working directory
cd /home/simon/bookrec

# Activate conda environment
source /home/simon/miniconda3/bin/activate bookrec-api

# Run the worker with logging
python -m app.semantic_index.embedding.worker \
    --mode incremental \
    --limit 20 \
    --tags-version v2 \
    >> logs/embedding_worker.log 2>&1

# Optional: Add timestamp to log
echo "Run completed at $(date)" >> logs/embedding_worker.log
