"""Configuration module for MLX Audio ASR Service.

This module provides a centralized configuration class with typed fields,
explicit defaults, and environment variable override support.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class for the ASR service.

    All parameters can be overridden via environment variables with the ASR_ prefix.
    For example, ASR_MAX_FILE_SIZE_MB=50 overrides max_file_size_mb to 50.

    Attributes:
        language: Language hint for transcription. None enables auto-detection.
        max_tokens: Maximum tokens for transcription output (safe upper bound).
        chunk_duration: VAD chunk size in seconds.
        prefill_step_size: MLX-audio specific prefill step size.
        max_file_size_mb: Maximum allowed audio file size in megabytes.
        max_audio_duration_sec: Maximum allowed audio duration in seconds.
        request_timeout_sec: Request timeout in seconds.
        queue_max_size: Maximum number of requests in the queue.
        worker_concurrency: Number of concurrent workers (single GPU = 1).
    """

    model_config = SettingsConfigDict(
        env_prefix="ASR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model path
    model_path: Path = Field(
        default=Path("/Volumes/AigoP3500/models/lmstudio/models/mlx-community/Qwen3-ASR-0.6B-bf16"),
        description="Path to the ASR model directory.",
    )

    # Language and model parameters
    language: Optional[str] = Field(
        default=None,
        description="Language hint for transcription. None enables auto-detection.",
    )

    max_tokens: int = Field(
        default=4800,
        ge=100,
        le=10000,
        description="Maximum tokens for transcription output.",
    )

    chunk_duration: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="VAD chunk size in seconds.",
    )

    prefill_step_size: int = Field(
        default=2400,
        ge=100,
        le=10000,
        description="MLX-audio specific prefill step size.",
    )

    # Input limits
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum allowed audio file size in megabytes.",
    )

    max_audio_duration_sec: float = Field(
        default=3600.0,
        ge=1.0,
        le=36000.0,
        description="Maximum allowed audio duration in seconds.",
    )

    # Supported audio formats
    supported_formats: list[str] = Field(
        default=["wav", "mp3", "m4a", "flac", "ogg"],
        description="Supported audio file formats.",
    )

    # Request handling
    request_timeout_sec: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Request timeout in seconds.",
    )

    queue_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of requests in the queue.",
    )

    worker_concurrency: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of concurrent workers (single GPU = 1).",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Normalize language parameter."""
        if v is None:
            return None
        normalized = v.strip()
        if not normalized:
            return None
        if normalized.lower() in {"auto", "none", "null"}:
            return None
        return normalized

    @field_validator("max_tokens", "prefill_step_size", "queue_max_size", "worker_concurrency")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


# Global config instance - can be imported and used throughout the application
config = Config()
