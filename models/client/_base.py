# models/client/_base.py
"""
Base HTTP client for model server communication.

Provides a shared asynccontextmanager-compatible lifecycle, a reusable
httpx.AsyncClient singleton with connection pooling, and centralized
translation of httpx exceptions into the model server exception hierarchy.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from models.client._exceptions import (
    ModelServerRequestError,
    ModelServerUnavailableError,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=2.0)


class BaseModelServerClient:
    """
    Async HTTP client base for a single model server.

    Subclasses declare a default base URL via the class attribute
    `_DEFAULT_BASE_URL_ENV` (the env var name) and `_SERVER_NAME` (for
    logging). The shared `httpx.AsyncClient` is created on first use and
    reused across requests for connection pooling.

    Lifecycle: call `await client.aclose()` on application shutdown, or use
    the FastAPI lifespan to manage teardown centrally.
    """

    _DEFAULT_BASE_URL_ENV: str = ""
    _SERVER_NAME: str = "unknown"

    def __init__(self, base_url: str, timeout: httpx.Timeout = _DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close the underlying httpx client and release connection pool resources."""
        await self._client.aclose()

    async def _post(self, path: str, body: Any) -> dict:
        """
        POST a Pydantic model as JSON and return the parsed response dict.

        Args:
            path: URL path relative to base_url (e.g. '/embed').
            body: Pydantic model instance; serialized via model_dump().

        Returns:
            Parsed JSON response as a dict.

        Raises:
            ModelServerUnavailableError: On connection failure, timeout, or 5xx.
            ModelServerRequestError: On 4xx response from the server.
        """
        try:
            response = await self._client.post(
                path,
                content=body.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = _extract_detail(exc.response)
            if status >= 500:
                logger.error(
                    "%s server error on %s: %s %s",
                    self._SERVER_NAME,
                    path,
                    status,
                    detail,
                )
                raise ModelServerUnavailableError(
                    f"{self._SERVER_NAME} returned {status}: {detail}"
                ) from exc
            else:
                logger.warning(
                    "%s request error on %s: %s %s",
                    self._SERVER_NAME,
                    path,
                    status,
                    detail,
                )
                raise ModelServerRequestError(
                    f"{self._SERVER_NAME} rejected request ({status}): {detail}"
                ) from exc

        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            logger.error(
                "%s unreachable at %s%s: %s",
                self._SERVER_NAME,
                self._base_url,
                path,
                exc,
            )
            raise ModelServerUnavailableError(f"{self._SERVER_NAME} unreachable: {exc}") from exc

        except httpx.RequestError as exc:
            logger.error(
                "%s request failed on %s: %s",
                self._SERVER_NAME,
                path,
                exc,
            )
            raise ModelServerUnavailableError(f"{self._SERVER_NAME} request failed: {exc}") from exc


def _extract_detail(response: httpx.Response) -> str:
    """
    Extract a human-readable error detail string from an HTTP error response.

    Attempts to parse a FastAPI-style JSON body with a 'detail' key, falling
    back to the raw response text if the body is not valid JSON.
    """
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text
