# app/enrichment/llm_client.py
import json
import logging
import os
import re
from typing import Any, Dict, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to reuse your shared factory if present (won't change the chat model)
try:
    from app.agents.settings import get_llm  # optional fallback
except Exception:
    get_llm = None  # pragma: no cover

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "Meta-Llama-3.1-8B-Instruct-Turbo")

# NEW: if set to "1", do NOT fall back to get_llm(); require DeepInfra to work
ENRICH_FORCE_DEEPINFRA = os.getenv("ENRICH_FORCE_DEEPINFRA", "0") == "1"


def _deepinfra_llm():
    """
    Build a ChatOpenAI-compatible client pointed at DeepInfra only for enrichment.
    This does NOT affect your chat agent.
    """
    from langchain_openai import ChatOpenAI

    if not DEEPINFRA_API_KEY:
        return None
    return ChatOpenAI(
        model=DEEPINFRA_MODEL,
        temperature=0,
        timeout=60,
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_BASE_URL,
    )


def _resolve_llm_for_enrichment():
    llm = _deepinfra_llm()
    if llm is None and ENRICH_FORCE_DEEPINFRA:
        # Force failure if DeepInfra key is missing or unusable
        raise RuntimeError("ENRICH_FORCE_DEEPINFRA=1 is set but DEEPINFRA_API_KEY is missing.")
    if llm is None:
        if get_llm is None:
            raise RuntimeError("No DeepInfra credentials and no get_llm() fallback available.")
        llm = get_llm()
    return llm


def ensure_enrichment_ready() -> None:
    """
    Fail fast before a long run. We do a tiny JSON echo call to verify
    auth/quota/routing is OK. If it fails, raise an exception so the caller
    can abort the job immediately.
    """
    llm = _resolve_llm_for_enrichment()
    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": 'Return exactly this JSON: {"ok": true}'},
                {"role": "user", "content": 'Respond with {"ok": true} and nothing else.'},
            ]
        )
        raw = getattr(resp, "content", resp)
        parsed = json.loads(raw)
        if not (isinstance(parsed, dict) and parsed.get("ok") is True):
            raise RuntimeError("Warmup check failed: unexpected response body.")
    except Exception as e:
        raise RuntimeError(f"LLM warmup failed (auth/quota/network): {e}")


def _extract_json(raw: str) -> str:
    """
    Extract JSON from LLM response, handling:
    - Clean JSON (expected after prompt improvements)
    - Markdown code blocks: ```json\n{...}\n```
    - Multiple JSON objects (returns the last valid one)
    - Explanatory text before/after JSON
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from LLM")

    raw = raw.strip()

    # Fast path: clean JSON (should be common after prompt fix)
    if raw.startswith("{") and raw.endswith("}"):
        try:
            json.loads(raw)
            return raw
        except:
            pass  # Fall through to more aggressive parsing

    # Extract from markdown code blocks
    json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if json_blocks:
        # Try each block from last to first (prefer later corrections)
        for block in reversed(json_blocks):
            block = block.strip()
            try:
                json.loads(block)
                return block
            except:
                continue

    # Find all JSON-like objects (balanced braces)
    # This handles "text before {json} text after" and multiple JSON objects
    depth = 0
    start = None
    candidates = []

    for i, char in enumerate(raw):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(raw[start : i + 1])
                start = None

    # Try candidates from last to first (prefer later corrections)
    for candidate in reversed(candidates):
        try:
            json.loads(candidate)
            return candidate
        except:
            continue

    raise ValueError(f"No valid JSON found in response (first 300 chars): {raw[:300]}")


def call_enrichment_llm(system: str, user: str) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    Calls the enrichment model and returns strict JSON.
    Priority:
      1) DeepInfra (if DEEPINFRA_API_KEY present)
      2) Shared get_llm() factory (fallback unless ENRICH_FORCE_DEEPINFRA=1)

    Returns:
        (parsed_json, usage_dict, latency_ms)
    """
    import time

    llm = _resolve_llm_for_enrichment()

    start = time.time()
    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    latency_ms = int((time.time() - start) * 1000)

    # Extract raw content
    raw = getattr(resp, "content", resp)

    if not raw or not isinstance(raw, str):
        logger.error(f"LLM returned non-string content: {type(raw)}")
        raise RuntimeError(f"LLM returned invalid response type: {type(raw)}")

    # Log first 200 chars for debugging
    logger.debug(f"Raw LLM response (first 200 chars): {raw[:200]}")

    # Extract JSON (handles markdown, extra text, etc.)
    try:
        json_str = _extract_json(raw)
    except ValueError as e:
        logger.error(f"Failed to extract JSON from response: {e}")
        logger.error(f"Full response: {raw}")
        raise RuntimeError(f"No JSON found in response: {str(e)}")

    # Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Attempted to parse: {json_str[:500]}")
        logger.error(f"Full raw response: {raw}")
        raise RuntimeError(f"Parse error: {e}")

    # Extract usage if available
    usage = {}
    if hasattr(resp, "usage_metadata"):
        usage = resp.usage_metadata
    elif hasattr(resp, "response_metadata"):
        usage = resp.response_metadata.get("usage", {})

    return parsed, usage, latency_ms
