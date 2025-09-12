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
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "Meta-Llama-3.1-8B-Instruct-Turbo")

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

def call_enrichment_llm(system: str, user: str) -> Dict[str, Any]:
    """
    Calls the enrichment model and returns strict JSON.
    Priority:
      1) DeepInfra (if DEEPINFRA_API_KEY present)
      2) Shared get_llm() factory (fallback)
    """
    llm = _deepinfra_llm()
    if llm is None:
        if get_llm is None:
            raise RuntimeError("No DeepInfra credentials and no get_llm() fallback available.")
        # Fallback uses your current default model; leaves the chat selection untouched.
        llm = get_llm()

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
