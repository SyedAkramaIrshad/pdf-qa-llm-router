"""Configuration settings using Pydantic for validation."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = DATA_DIR / "indices"
PDFS_DIR = DATA_DIR / "pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # GLM API Configuration
    glm_api_key: str = Field(default="", alias="GLM_API_KEY")
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        alias="GLM_BASE_URL"
    )
    # Using FREE models for POC (no billing/credits required)
    # Text: glm-4.5-flash (free, stable text model)
    # Vision: glm-4.6v-flash (free multimodal model)
    glm_model: str = Field(default="glm-4.5-flash", alias="GLM_MODEL")
    glm_vision_model: str = Field(default="glm-4.6v-flash", alias="GLM_VISION_MODEL")

    # Processing Configuration
    chunk_size: int = Field(default=10, alias="CHUNK_SIZE")
    # Lite plan has ~3 concurrent connection limit
    max_concurrent_calls: int = Field(default=3, alias="MAX_CONCURRENT_CALLS")
    max_retry_attempts: int = Field(default=3, alias="MAX_RETRY_ATTEMPTS")

    # Indexing Configuration
    # Number of sections to process in parallel during indexing (default: 1 = sequential)
    # Set to >1 for parallel processing (may hit rate limits on free tier)
    indexing_concurrent: int = Field(default=1, alias="INDEXING_CONCURRENT")
    # Delay between API calls in seconds (helps avoid rate limits)
    api_delay: float = Field(default=1.0, alias="API_DELAY")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Directory Paths (computed)
    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=DATA_DIR)
    indices_dir: Path = Field(default=INDICES_DIR)
    pdfs_dir: Path = Field(default=PDFS_DIR)
    extracted_dir: Path = Field(default=EXTRACTED_DIR)

    @field_validator("glm_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty."""
        if not v:
            raise ValueError(
                "GLM_API_KEY is required. Set it in .env file or as environment variable."
            )
        return v

    @field_validator("chunk_size", "max_concurrent_calls", "max_retry_attempts")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that numeric settings are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure directories exist
        _settings.indices_dir.mkdir(parents=True, exist_ok=True)
        _settings.pdfs_dir.mkdir(parents=True, exist_ok=True)
        _settings.extracted_dir.mkdir(parents=True, exist_ok=True)
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
