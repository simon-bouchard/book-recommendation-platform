# app/agents/settings.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, Optional

class ChatSettings(BaseSettings):
    chat_require_login: bool = False
    redis_url: str = "redis://localhost:6379/0"
    chat_ttl_sec: int = 172800
    chat_hist_turns: int = 3
    chat_limits_per_min_user: int = 5
    chat_limits_per_day_user: int = 40
    chat_limits_per_min_fallback: int = 3
    chat_limits_per_day_fallback: int = 10
    chat_limits_per_day_system: int = 250
    chat_local_tz: str = "America/Toronto"

    # LLM selection (provider decided by environment)
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepinfra")  # "deepinfra" | "openai"

    # OpenAI-compatible settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")  # optional override

    # DeepInfra (OpenAI-compatible API)
    deepinfra_api_key: Optional[str] = os.getenv("DEEPINFRA_API_KEY")
    deepinfra_base_url: str = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")

    llm_model_small: str = os.getenv("LLM_MODEL_SMALL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    llm_model_medium: str = os.getenv("LLM_MODEL_MEDIUM", "meta-llama/Meta-Llama-3.1-70B-Instruct")
    llm_model_large: str = os.getenv("LLM_MODEL_LARGE", "openai/gpt-4")

    # Embedding hook (callable: List[str] -> np.ndarray [n, d])
    embedder: Any = None

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore",
        arbitrary_types_allowed=True,
    )

# ---- Embedding factory ----
from sentence_transformers import SentenceTransformer
import numpy as np

def _make_default_embedder():
    """
    Returns a callable: List[str] -> np.ndarray [n, d]
    Uses a cached sentence-transformers model by default.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _emb(texts: list[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True).astype("float32")

    return _emb

settings = ChatSettings()
settings.embedder = _make_default_embedder()

# ---- LLM provider + factory ----
from langchain_openai import ChatOpenAI

def get_provider() -> Dict[str, Any]:
    """
    Resolve the active provider from environment-backed settings.
    Returns a dict with normalized connection/config values.
    """
    provider = (settings.llm_provider or "deepinfra").strip().lower()

    if provider == "deepinfra":
        return {
            "name": "deepinfra",
            "base_url": settings.deepinfra_base_url,
            "api_key": settings.deepinfra_api_key,
            "default_model": settings.llm_model_medium,
        }

    # Fallback: OpenAI
    return {
        "name": "openai",
        "base_url": settings.openai_base_url,  # may be None (uses library default)
        "api_key": settings.openai_api_key,
        "default_model": settings.llm_model_medium,
    }

def get_model_for_tier(tier: str) -> str:
    t = (tier or "").strip().lower()
    if t == "small":
        return settings.llm_model_small
    if t == "medium":
        return settings.llm_model_medium
    if t == "large":
        return settings.llm_model_large
    return settings.llm_model_medium

def get_llm(
    model: Optional[str] = None,
    *,
    tier: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 30,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    json_mode: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Return a LangChain Chat model for the active provider.
    Auto-routes provider if the selected model clearly belongs to OpenAI.
    """
    p = get_provider()

    # Resolve explicit -> tier -> provider default
    resolved = model or get_model_for_tier(tier) or p.get("default_model")
    if not resolved:
        raise RuntimeError(
            "No LLM model configured. Set model=..., or LLM_MODEL_SMALL/MEDIUM/LARGE, "
            "or provider.default_model."
        )

    # Important: make it a real string, NOT the class `str`
    selected_model = str(resolved)

    # Defensive guard: blow up early if something is off
    if not isinstance(selected_model, str) or selected_model is str:
        raise TypeError(f"Resolved model must be a string, got: {type(resolved)!r} ({resolved!r})")

    # Identify if this looks like an OpenAI-family model
    wants_openai = selected_model.startswith(("gpt-", "o3", "openai/"))

    provider_name = p["name"]
    base_url = p["base_url"]
    api_key = p["api_key"]

    if provider_name == "deepinfra" and wants_openai:
        # Try to switch to OpenAI if possible
        if settings.openai_api_key:
            provider_name = "openai"
            base_url = settings.openai_base_url  # can be None (uses SDK default)
            api_key = settings.openai_api_key

            # Normalize common openai/* prefixes
            if selected_model.startswith("openai/"):
                selected_model = selected_model.split("/", 1)[1]
        else:
            # No OpenAI key, stay on DeepInfra but pick a valid DeepInfra model
            selected_model = p["default_model"]

    # Assemble base args
    init_kwargs: Dict[str, Any] = {
        "model": selected_model,
        "temperature": float(temperature),
        "timeout": int(timeout),
        "api_key": api_key,
        "base_url": base_url,
    }

    if max_tokens is not None:
        init_kwargs["max_tokens"] = int(max_tokens)
    if seed is not None:
        init_kwargs["seed"] = int(seed)

    # JSON (if backend supports it; harmless if ignored)
    if json_mode:
        init_kwargs.setdefault("model_kwargs", {})
        init_kwargs["model_kwargs"]["response_format"] = {"type": "json_object"}

    if model_kwargs:
        mk = dict(init_kwargs.get("model_kwargs", {}))
        mk.update(model_kwargs)
        init_kwargs["model_kwargs"] = mk

    # Debug prints (show effective config)
    print(f"DEBUG get_llm: provider={provider_name}")
    print(f"DEBUG get_llm: model={init_kwargs['model']}")
    print(f"DEBUG get_llm: base_url={init_kwargs.get('base_url') or 'sdk_default'}")
    print(f"DEBUG get_llm: api_key_present={bool(init_kwargs.get('api_key'))}")

    return ChatOpenAI(**init_kwargs)
