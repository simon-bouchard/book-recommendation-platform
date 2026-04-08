# Location: app/semantic_index/embedding/embedding_client.py
# Wrapper for sentence-transformers with retry logic and batching

import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingClient:
    """
    Wrapper for sentence-transformers model with retry logic.

    Handles:
    - Model loading and caching
    - Batch encoding
    - Retry on failures
    - Progress logging
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()

    def encode_batch(
        self, texts: List[str], show_progress: bool = False, max_retries: int = 3
    ) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.

        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            max_retries: Number of retry attempts on failure

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        for attempt in range(max_retries):
            try:
                # Use model's batch encoding with normalization
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # L2 normalize for cosine similarity
                )

                return embeddings.astype(np.float32)

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # exponential backoff
                    print(f"Encoding failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Encoding failed after {max_retries} attempts: {e}")
                    raise

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (convenience method)"""
        return self.encode_batch([text])[0]

    def build_embedding_text(self, item: dict, tone_names: List[str], genre_name: str) -> str:
        """
        Build the text string that will be embedded.

        Format: "{title} — {author} | subjects: {subjects} | tones: {tones} | genre: {genre} | vibe: {vibe}"

        This matches the format from the enrichment plan.
        """
        title = item.get("title", "").strip()
        author = item.get("author", "").strip()
        subjects = ", ".join(item.get("subjects", []))
        tones = ", ".join(tone_names)
        vibe = item.get("vibe", "").strip()

        # Build compact but information-rich text
        parts = [f"{title} — {author}"]

        if subjects:
            parts.append(f"subjects: {subjects}")

        if tones:
            parts.append(f"tones: {tones}")

        if genre_name:
            parts.append(f"genre: {genre_name}")

        if vibe:
            parts.append(f"vibe: {vibe}")

        return " | ".join(parts)
