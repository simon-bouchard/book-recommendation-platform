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

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore",
    )
    
settings = ChatSettings()
