# ops/training/reload_signal.py
"""
Triggers a graceful Gunicorn reload across all model server containers.

Sends SIGHUP to each container via `docker kill --signal=SIGHUP`. Because
the Dockerfile uses `exec gunicorn`, Gunicorn is PID 1 and receives the
signal directly. Gunicorn re-executes itself with preload_app=True, loads
fresh artifacts in the master process, forks new copy-on-write workers,
then drains and terminates the old generation.

This is the correct reload primitive. Per-worker in-process reloads defeat
copy-on-write sharing because each worker allocates its own private copy of
every artifact array.
"""

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINERS = [
    "model-server-embedder",
    "model-server-similarity",
    "model-server-als",
    "model-server-metadata",
]

_ENV_VAR = "MODEL_SERVER_CONTAINERS"


def _resolve_containers() -> list[str]:
    """
    Return the list of container names to signal.

    Reads MODEL_SERVER_CONTAINERS from the environment if set, treating its
    value as a comma-separated list. Falls back to the hardcoded defaults
    matching the names defined in docker-compose.yml.
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    if raw:
        return [name.strip() for name in raw.split(",") if name.strip()]
    return list(_DEFAULT_CONTAINERS)


def signal_workers_reload(docker_executable: Optional[str] = None) -> None:
    """
    Send SIGHUP to all model server containers to trigger a graceful reload.

    Each container is signalled independently. A failure on one container
    (e.g. it is not running) is logged as an error but does not prevent the
    remaining containers from being signalled.

    Args:
        docker_executable: Path to the docker binary. Defaults to 'docker',
                           which resolves via PATH. Exposed for testing.

    Raises:
        RuntimeError: If every container signal attempt failed, indicating a
                      systemic problem (docker not available, wrong host, etc.).
    """
    docker = docker_executable or "docker"
    containers = _resolve_containers()

    logger.info(
        "Sending SIGHUP to %d model server container(s): %s",
        len(containers),
        ", ".join(containers),
    )

    failures: list[str] = []

    for container in containers:
        try:
            result = subprocess.run(
                [docker, "kill", "--signal=SIGHUP", container],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("SIGHUP sent to container '%s'", container)
            else:
                msg = result.stderr.strip() or result.stdout.strip()
                logger.error(
                    "Failed to send SIGHUP to container '%s': %s",
                    container,
                    msg,
                )
                failures.append(container)

        except subprocess.TimeoutExpired:
            logger.error("Timed out signalling container '%s'", container)
            failures.append(container)
        except FileNotFoundError:
            logger.error(
                "docker executable not found at '%s'. Ensure docker is installed and on PATH.",
                docker,
            )
            failures.append(container)

    if len(failures) == len(containers):
        raise RuntimeError(
            f"SIGHUP failed for all {len(containers)} container(s). "
            "Gunicorn workers have not been reloaded. "
            f"Failures: {failures}"
        )

    if failures:
        logger.warning(
            "SIGHUP failed for %d of %d container(s): %s. "
            "Those servers are still running stale artifacts.",
            len(failures),
            len(containers),
            failures,
        )
