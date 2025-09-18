# app/enrichment/llm_client.py
from typing import Dict, Any
import os, json

# Try to reuse your shared factory if present (won't change the chat model)
try:
    from app.agents.settings import get_llm  # optional fallback
except Exception:
    get_llm = None  # pragma: no cover

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "Meta-Llama-3.1-70B-Instruct-Turbo")

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
        resp = llm.invoke([
            {"role": "system", "content": "Return exactly this JSON: {\"ok\": true}"},
            {"role": "user", "content": "Respond with {\"ok\": true} and nothing else."},
        ])
        raw = getattr(resp, "content", resp)
        parsed = json.loads(raw)
        if not (isinstance(parsed, dict) and parsed.get("ok") is True):
            raise RuntimeError("Warmup check failed: unexpected response body.")
    except Exception as e:
        raise RuntimeError(f"LLM warmup failed (auth/quota/network): {e}")

def call_enrichment_llm(system: str, user: str) -> Dict[str, Any]:
    """
    Calls the enrichment model and returns strict JSON.
    Priority:
      1) DeepInfra (if DEEPINFRA_API_KEY present)
      2) Shared get_llm() factory (fallback unless ENRICH_FORCE_DEEPINFRA=1)
    """
    llm = _resolve_llm_for_enrichment()
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    # Expect strict JSON
    raw = getattr(resp, "content", resp)
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Parse error: {e}")
