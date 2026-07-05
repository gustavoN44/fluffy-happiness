"""Application configuration, loaded once from the environment / .env.

A single typed settings object so config has one validated source of truth
instead of scattered os.getenv calls. Missing required values (e.g. the OpenAI
key) fail loudly at import time rather than deep inside an API call.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Read .env; ignore the POSTGRES_* vars there that belong to docker-compose.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required — no default, so startup fails if absent.
    openai_api_key: str
    database_url: str

    # Phase 1 baseline embedding config. Hardcoded here now; becomes
    # config-swappable in Phase 3. embedding_dim MUST match the model's output
    # and the chunks.embedding column (vector(1536)).
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Phase 1 baseline generation config. Low temperature for grounded,
    # reproducible answers. Swappable in later phases; held constant in Phase 5.
    generation_model: str = "gpt-4o-mini"
    generation_temperature: float = 0.0


settings = Settings()
