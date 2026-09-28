"""Runtime settings, read from environment variables (or a local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ClinicalTrials.gov
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    ctgov_timeout_s: float = 30.0
    ctgov_max_retries: int = 4
    # The API asks for roughly <= 50 requests/minute. We space requests out rather
    # than waiting to be throttled.
    ctgov_min_interval_s: float = 1.2
    ctgov_page_size: int = 1000  # the API's hard maximum

    # On-disk response cache. Also lets the service replay recorded responses
    # (CTGOV_OFFLINE=true), which is how tests and example runs stay reproducible.
    cache_dir: Path = Path(".cache/ctgov")
    cache_ttl_s: int = 24 * 3600
    ctgov_offline: bool = False

    # Planner (OpenAI). Without a key the rule-based planner is used.
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    planner_timeout_s: float = 30.0
    planner_max_attempts: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
