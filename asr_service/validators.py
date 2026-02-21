"""Input validation module for MLX Audio ASR Service.

Provides validation functions for:
- File size validation
- Audio format validation
- Audio duration validation using ffprobe
- Request timeout handling
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

from asr_service.config import config


# Supported MIME types mapping
SUPPORTED_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


class ValidationError(Exception):
    """Base exception for validation errors."""

    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class FileSizeError(ValidationError):
    """Exception for file size validation errors (HTTP 413)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=413)


class FormatError(ValidationError):
    """Exception for format validation errors (HTTP 415)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=415)


class DurationError(ValidationError):
    """Exception for duration validation errors (HTTP 400)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class TimeoutError(ValidationError):
    """Exception for request timeout errors (HTTP 408)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=408)


def validate_file_size(content: bytes) -> None:
    """Validate that file content does not exceed max file size.

    Args:
        content: Raw file bytes.

    Raises:
        FileSizeError: If file exceeds max_file_size_mb config.
    """
    max_bytes = config.max_file_size_mb * 1024 * 1024
    actual_size = len(content)

    if actual_size > max_bytes:
        raise FileSizeError(
            f"File size {actual_size / (1024 * 1024):.2f} MB exceeds "
            f"maximum allowed size of {config.max_file_size_mb} MB"
        )


def validate_format(filename: str, content: bytes) -> None:
    """Validate that file format is supported.

    Args:
        filename: Name of the uploaded file.
        content: Raw file bytes for magic number validation.

    Raises:
        FormatError: If format is not supported.
    """
    # Extract extension from filename
    file_path = Path(filename)
    extension = file_path.suffix.lstrip(".").lower()

    # Check extension against whitelist
    if extension not in config.supported_formats:
        raise FormatError(
            f"Unsupported audio format: '{extension}'. "
            f"Supported formats: {', '.join(config.supported_formats)}"
        )

    # Validate magic numbers for common formats
    if extension == "wav":
        # WAV files start with "RIFF" header
        if not content.startswith(b"RIFF"):
            raise FormatError("Invalid WAV file: missing RIFF header")
    elif extension == "mp3":
        # MP3 files start with ID3 tag or sync word
        if not (content.startswith(b"ID3") or content[:2] == b"\xff\xfb"):
            raise FormatError("Invalid MP3 file: missing ID3 or sync header")
    elif extension == "flac":
        # FLAC files start with "fLaC"
        if not content.startswith(b"fLaC"):
            raise FormatError("Invalid FLAC file: missing fLaC header")
    elif extension == "ogg":
        # OGG files start with "OggS"
        if not content.startswith(b"OggS"):
            raise FormatError("Invalid OGG file: missing OggS header")
    # m4a doesn't have reliable magic number, skip additional validation


def get_audio_duration_ffprobe(audio_path: Path) -> float:
    """Get audio duration using ffprobe.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        duration = float(result.stdout.strip())
        return duration

    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found - please install ffmpeg")
    except ValueError:
        raise RuntimeError("Could not parse duration from ffprobe output")


def validate_duration(audio_path: Path) -> None:
    """Validate that audio duration does not exceed max allowed.

    Args:
        audio_path: Path to the audio file.

    Raises:
        DurationError: If audio duration exceeds max_audio_duration_sec.
    """
    try:
        duration = get_audio_duration_ffprobe(audio_path)

        if duration > config.max_audio_duration_sec:
            raise DurationError(
                f"Audio duration {duration:.2f}s exceeds maximum allowed "
                f"duration of {config.max_audio_duration_sec}s"
            )

    except RuntimeError as e:
        # If we can't determine duration, log warning but allow through
        # This prevents blocking valid files due to ffprobe issues
        import logging

        logging.getLogger("mlx_asr").warning(
            f"Could not determine audio duration: {e}. Allowing file through."
        )


def validate_input(
    filename: str,
    content: bytes,
    audio_path: Optional[Path] = None,
) -> None:
    """Run all input validations in order.

    Args:
        filename: Name of the uploaded file.
        content: Raw file bytes.
        audio_path: Optional path to temp file for duration check.

    Raises:
        FileSizeError: If file is too large (HTTP 413).
        FormatError: If format is unsupported (HTTP 415).
        DurationError: If duration exceeds limit (HTTP 400).
    """
    # 1. File size validation
    validate_file_size(content)

    # 2. Format validation
    validate_format(filename, content)

    # 3. Duration validation (if we have a path)
    if audio_path is not None:
        validate_duration(audio_path)


class TimeoutContext:
    """Context manager for request timeout handling."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def check_timeout(self) -> None:
        """Check if request has exceeded timeout.

        Raises:
            TimeoutError: If request has timed out.
        """
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(
                f"Request exceeded timeout of {self.timeout_seconds}s "
                f"(elapsed: {elapsed:.2f}s)"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
