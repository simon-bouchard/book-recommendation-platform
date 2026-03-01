# models/core/reload_poller.py
"""
Background polling service for model reloading across multiple workers.
Monitors signal file for timestamp changes and triggers reload on all workers.
"""

import asyncio
import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SIGNAL_FILE = Path(__file__).parent.parent.parent / "models" / "data" / ".reload_signal"


class ModelReloadPoller:
    """
    Polls signal file every 30 seconds to detect model updates.

    Uses timestamp-based signaling to coordinate reloads across multiple workers.
    Safe to run in multiple workers - each tracks its own last reload timestamp.

    The signal file path is resolved from the MODEL_RELOAD_SIGNAL_FILE environment
    variable when set, falling back to the default path relative to this file.
    Set the environment variable in container deployments where the project is
    mounted at a different path than the development machine.
    """

    SIGNAL_FILE = Path(os.environ.get("MODEL_RELOAD_SIGNAL_FILE", str(_DEFAULT_SIGNAL_FILE)))
    POLL_INTERVAL = 30.0

    def __init__(self):
        self._last_reload_timestamp = 0.0
        self._running = False
        self._task = None

    async def start(self):
        """Start the background polling task."""
        if self._running:
            logger.warning("Model reload poller already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Model reload poller started (polling every %.0fs, signal file: %s)",
            self.POLL_INTERVAL,
            self.SIGNAL_FILE,
        )

    async def stop(self):
        """Stop the background polling task gracefully."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Model reload poller stopped")

    async def _poll_loop(self):
        """Main polling loop - checks signal file every 30 seconds."""
        while self._running:
            try:
                await self._check_and_reload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model reload poller: {e}", exc_info=True)

            await asyncio.sleep(self.POLL_INTERVAL)

    async def _check_and_reload(self):
        """Check signal file and reload if timestamp is newer."""
        if not self.SIGNAL_FILE.exists():
            return

        try:
            with open(self.SIGNAL_FILE, "r") as f:
                content = f.read().strip()

            if not content:
                return

            signal_timestamp = float(content)

            if signal_timestamp > self._last_reload_timestamp:
                logger.info(
                    "Detected model update signal (%.3f > %.3f), reloading models...",
                    signal_timestamp,
                    self._last_reload_timestamp,
                )

                start_time = time.time()
                await self._reload_models()
                elapsed_ms = (time.time() - start_time) * 1000

                self._last_reload_timestamp = signal_timestamp

                logger.info(
                    "Models reloaded successfully in %.0fms (timestamp: %.3f)",
                    elapsed_ms,
                    signal_timestamp,
                )

        except ValueError as e:
            logger.error(f"Invalid timestamp in signal file: {e}")
        except Exception as e:
            logger.error(f"Failed to check/reload models: {e}", exc_info=True)

    async def _reload_models(self):
        """Reload all model artifacts and clear caches."""
        from models.data.loaders import clear_cache, preload_all_artifacts
        from models.infrastructure.subject_embedder import SubjectEmbedder
        from models.infrastructure.als_model import ALSModel
        from models.infrastructure.similarity_indices import reset_indices
        from models.infrastructure.subject_scorer import SubjectScorer
        from models.infrastructure.hybrid_scorer import HybridScorer
        from models.infrastructure.popularity_scorer import PopularityScorer
        from models.cache import clear_ml_cache

        clear_cache()
        SubjectEmbedder.reset()
        ALSModel.reset()
        reset_indices()
        SubjectScorer.reset()
        HybridScorer.reset()
        PopularityScorer.reset()

        clear_ml_cache()

        preload_all_artifacts()


_poller_instance = None


def get_poller() -> ModelReloadPoller:
    """Get or create the global poller instance."""
    global _poller_instance
    if _poller_instance is None:
        _poller_instance = ModelReloadPoller()
    return _poller_instance
