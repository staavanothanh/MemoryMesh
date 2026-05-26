"""EmbeddingFactory — create embedding provider based on config.

Uses factory pattern to keep MCP server code agnostic of the
actual embedding backend. Graceful fallback: remote → local → error.
"""

import logging
from typing import Optional

from ..config import EmbeddingConfig
from .providers import EmbeddingProvider, LocalEmbeddingProvider, RemoteEmbeddingProvider

logger = logging.getLogger(__name__)


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create embedding provider based on configuration.

    Resolution order:
    1. If mode=remote and remote_api_url is set → RemoteEmbeddingProvider
    2. If mode=local (default) → LocalEmbeddingProvider
    3. If mode=remote but no API configured → fallback to LocalEmbeddingProvider
    """
    if config.mode == "remote" and config.remote_api_url:
        logger.info("Creating RemoteEmbeddingProvider (url=%s)", config.remote_api_url)
        return RemoteEmbeddingProvider(config)

    try:
        import sentence_transformers  # noqa: F401
        logger.info("Creating LocalEmbeddingProvider (model=%s)", config.model)
        return LocalEmbeddingProvider(config.model)
    except ImportError:
        if config.mode == "remote":
            raise RuntimeError(
                "Remote embedding API not configured and local embedding not available. "
                "Set REMOTE_EMBEDDING_API_URL or install: pip install 'memorymesh[local]'"
            )
        raise RuntimeError(
            "Local embedding requires sentence-transformers. "
            "Install with: pip install 'memorymesh[local]'"
        )


def has_local_embedding() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
