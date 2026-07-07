"""
Centralized application settings.
Loaded once and reused everywhere via dependency injection.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = "development"
    gemini_api_key: str = "YOUR_ACTUAL_GEMINI_API_KEY"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()