from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    elasticsearch_url: Optional[str] = None
    elasticsearch_api_key: Optional[str] = None
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
