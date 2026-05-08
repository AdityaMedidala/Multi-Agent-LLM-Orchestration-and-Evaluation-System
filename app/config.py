from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- External API keys ---
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str
    cohere_api_key: str

    # --- Database ---
    # asyncpg driver used by SQLAlchemy async engine and FastAPI
    database_url: str
    # psycopg2 driver used by Celery beat / worker (sync)
    database_url_sync: str

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- LangSmith tracing (optional) ---
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "mega-ai"

    # --- Logging ---
    log_level: str = "INFO"

    # --- Agent budget ---
    max_agent_budget_tokens: int = 8000  # default per-agent token budget

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
