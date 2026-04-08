# Location: app/semantic_index/embedding/accumulator_writer.py
# Writes embedding batches to NPZ files for incremental accumulation

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class AccumulatorWriter:
    """
    Writes embeddings to disk in NPZ batch files.

    Each batch contains:
    - embeddings: (N, D) float32 array
    - item_indices: (N,) int64 array
    - metadata: list of dicts with enrichment data

    Files are written to: {output_dir}/accumulator/batch_{batch_num:06d}.npz
    """

    def __init__(self, output_dir: str = "models/data/enriched_v2"):
        self.output_dir = Path(output_dir)
        self.accumulator_dir = self.output_dir / "accumulator"
        self.accumulator_dir.mkdir(parents=True, exist_ok=True)

        # Track current batch number
        self.current_batch_num = self._get_next_batch_num()

    def _get_next_batch_num(self) -> int:
        """Find the highest existing batch number and increment"""
        existing_batches = list(self.accumulator_dir.glob("batch_*.npz"))

        if not existing_batches:
            return 1

        # Extract batch numbers from filenames
        batch_nums = []
        for path in existing_batches:
            try:
                num = int(path.stem.split("_")[1])
                batch_nums.append(num)
            except (IndexError, ValueError):
                continue

        return max(batch_nums, default=0) + 1

    def write_batch(self, embeddings: np.ndarray, item_indices: List[int], metadata: List[Dict]):
        """
        Write a batch of embeddings to disk.

        Args:
            embeddings: (N, D) array of embeddings
            item_indices: List of item_idx values
            metadata: List of dicts with enrichment data (title, author, subjects, etc.)
        """
        if len(embeddings) != len(item_indices) != len(metadata):
            raise ValueError(
                f"Length mismatch: embeddings={len(embeddings)}, "
                f"indices={len(item_indices)}, metadata={len(metadata)}"
            )

        if len(embeddings) == 0:
            print("Warning: Skipping empty batch")
            return

        # Convert to numpy arrays
        item_indices_array = np.array(item_indices, dtype=np.int64)

        # Write NPZ file
        batch_path = self.accumulator_dir / f"batch_{self.current_batch_num:06d}.npz"

        np.savez_compressed(
            batch_path,
            embeddings=embeddings.astype(np.float32),
            item_indices=item_indices_array,
            metadata=json.dumps(metadata),  # Store as JSON string in NPZ
        )

        print(
            f"Wrote batch {self.current_batch_num}: {len(embeddings)} embeddings to {batch_path.name}"
        )
        self.current_batch_num += 1

    def load_batch(self, batch_num: int) -> tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load a specific batch (mainly for testing/debugging).

        Returns:
            (embeddings, item_indices, metadata)
        """
        batch_path = self.accumulator_dir / f"batch_{batch_num:06d}.npz"

        if not batch_path.exists():
            raise FileNotFoundError(f"Batch not found: {batch_path}")

        data = np.load(batch_path, allow_pickle=True)

        embeddings = data["embeddings"]
        item_indices = data["item_indices"]
        metadata = json.loads(str(data["metadata"]))

        return embeddings, item_indices, metadata

    def get_all_batches(self) -> List[Path]:
        """Get paths to all batch files, sorted by batch number"""
        batches = sorted(self.accumulator_dir.glob("batch_*.npz"))
        return batches

    def get_batch_count(self) -> int:
        """Count total batch files"""
        return len(list(self.accumulator_dir.glob("batch_*.npz")))

    def consolidate_batches(self) -> tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load and concatenate all batches into single arrays.
        Used when building the final FAISS index.

        Returns:
            (all_embeddings, all_item_indices, all_metadata)
        """
        batches = self.get_all_batches()

        if not batches:
            return np.array([]), np.array([]), []

        print(f"Consolidating {len(batches)} batches...")

        all_embeddings = []
        all_item_indices = []
        all_metadata = []

        for batch_path in batches:
            data = np.load(batch_path, allow_pickle=True)
            all_embeddings.append(data["embeddings"])
            all_item_indices.append(data["item_indices"])
            all_metadata.extend(json.loads(str(data["metadata"])))

        # Concatenate arrays
        embeddings = np.concatenate(all_embeddings, axis=0)
        item_indices = np.concatenate(all_item_indices, axis=0)

        print(f"Consolidated {len(embeddings)} total embeddings")

        return embeddings, item_indices, all_metadata
