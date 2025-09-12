# app/agents/settings.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # LLM selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepinfra")  # "deepinfra" | "openai"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")  # optional override

    deepinfra_model: str = os.getenv("DEEPINFRA_MODEL", "Meta-Llama-3.1-8B-Instruct-Turbo")
    deepinfra_api_key: str | None = os.getenv("DEEPINFRA_API_KEY")
    deepinfra_base_url: str = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore",
    )

settings = ChatSettings()

# ---- LLM factory ----
from langchain_openai import ChatOpenAI

def get_llm(model: str | None = None):
    """
    Return a LangChain Chat model pointed at the selected provider.
    Defaults to DeepInfra if LLM_PROVIDER=deepinfra.
    """
    provider = (settings.llm_provider or "deepinfra").lower()

    if provider == "deepinfra":
        return ChatOpenAI(
            model=model or settings.deepinfra_model,
            temperature=0,
            timeout=30,
            api_key=settings.deepinfra_api_key,
            base_url=settings.deepinfra_base_url,
        )

    # Fallback: OpenAI-compatible defaults
    return ChatOpenAI(
        model=model or settings.openai_model,
        temperature=0,
        timeout=30,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,  # can be None
    )
