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

    llm_model_small: str = os.getenv("LLM_MODEL_SMALL", "Meta-Llama-3.1-8B-Instruct-Turbo")
    llm_model_medium: str = os.getenv("LLM_MODEL_MEDIUM", "Meta-Llama-3.1-70B-Instruct-Turbo")
    llm_model_large: str = os.getenv("LLM_MODEL_LARGE", "gpt-4o")

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
    - Provider is decided by env (LLM_PROVIDER).
    - Agents can override model + tuning at construction time via kwargs.
    """
    p = get_provider()
    selected_model = model or (get_model_for_tier(tier) if tier else p["default_model"])

    # Assemble base args; keep explicit knobs first, extras under model_kwargs.
    init_kwargs: Dict[str, Any] = {
        "model": selected_model,
        "temperature": float(temperature),
        "timeout": int(timeout),
        "api_key": p["api_key"],
        "base_url": p["base_url"],
    }

    # Optional common knobs
    if max_tokens is not None:
        init_kwargs["max_tokens"] = int(max_tokens)
    if seed is not None:
        init_kwargs["seed"] = int(seed)

    # JSON bias (for router/structured outputs)
    if json_mode:
        # ChatOpenAI accepts response_format for JSON responses in OpenAI-compatible SDKs.
        # If unsupported by the backend, it will be ignored safely.
        init_kwargs.setdefault("model_kwargs", {})
        init_kwargs["model_kwargs"]["response_format"] = {"type": "json_object"}

    # Merge caller-supplied model_kwargs last (takes precedence within model_kwargs only)
    if model_kwargs:
        mk = dict(init_kwargs.get("model_kwargs", {}))
        mk.update(model_kwargs)
        init_kwargs["model_kwargs"] = mk

    return ChatOpenAI(**init_kwargs)
