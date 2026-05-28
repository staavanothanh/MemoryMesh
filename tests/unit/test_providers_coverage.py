"""Test uncovered paths in embeddings/providers.py:
NoneEmbeddingProvider, RemoteEmbeddingProvider edge cases, LocalEmbeddingProvider edge cases."""

import pytest
from unittest.mock import patch, AsyncMock, Mock, MagicMock

from memorymesh.embeddings.providers import (
    NoneEmbeddingProvider,
    LocalEmbeddingProvider,
    RemoteEmbeddingProvider,
    EmbeddingProvider,
)
from memorymesh.config import EmbeddingConfig


# ── NoneEmbeddingProvider ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_none_provider_get_embedding_returns_zeros():
    provider = NoneEmbeddingProvider()
    emb = await provider.get_embedding("anything")
    assert len(emb) == 384
    assert all(v == 0.0 for v in emb)


@pytest.mark.asyncio
async def test_none_provider_get_dimension_returns_384():
    provider = NoneEmbeddingProvider()
    dim = await provider.get_dimension()
    assert dim == 384


@pytest.mark.asyncio
async def test_none_provider_prewarm_does_nothing():
    provider = NoneEmbeddingProvider()
    await provider.prewarm()  # Should not raise
    emb = await provider.get_embedding("test")
    assert len(emb) == 384


# ── EmbeddingProvider base class ────────────────────────────────────────


class _ConcreteProvider(EmbeddingProvider):
    """Minimal concrete implementation for testing base class methods."""

    async def get_embedding(self, text: str):
        return [0.1, 0.2, 0.3]

    async def prewarm(self):
        pass


@pytest.mark.asyncio
async def test_base_provider_get_dimension_default():
    """Base EmbeddingProvider.get_dimension defaults to len(get_embedding('ping'))."""
    provider = _ConcreteProvider()
    dim = await provider.get_dimension()
    assert dim == 3


@pytest.mark.asyncio
async def test_base_provider_close_default():
    """Base EmbeddingProvider.close() is a no-op."""
    provider = _ConcreteProvider()
    await provider.close()  # Should not raise


# ── LocalEmbeddingProvider edge cases ──────────────────────────────────


@pytest.mark.asyncio
async def test_local_provider_get_embedding_not_initialized():
    """get_embedding raises RuntimeError if prewarm not called."""
    provider = LocalEmbeddingProvider("test-model")
    with pytest.raises(RuntimeError, match="not initialized"):
        await provider.get_embedding("test")


@pytest.mark.asyncio
async def test_local_provider_get_dimension_caches():
    """get_dimension caches after first call."""
    provider = LocalEmbeddingProvider("test-model")

    # Mock the internal model with a MagicMock so .encode().tolist() works
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1] * 128
    provider._model = mock_model

    dim = await provider.get_dimension()
    assert dim == 128
    # Second call uses cache
    dim2 = await provider.get_dimension()
    assert dim2 == 128


@pytest.mark.asyncio
async def test_local_provider_prewarm_missing_sentence_transformers():
    """prewarm raises RuntimeError with helpful message when sentence-transformers missing."""
    import builtins
    provider = LocalEmbeddingProvider("test-model")
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
            await provider.prewarm()


# ── RemoteEmbeddingProvider edge cases ─────────────────────────────────


@pytest.mark.asyncio
async def test_remote_provider_get_embedding_not_initialized():
    """get_embedding raises RuntimeError if prewarm not called."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)
    with pytest.raises(RuntimeError, match="not initialized"):
        await provider.get_embedding("test")


@pytest.mark.asyncio
async def test_remote_provider_get_dimension_caches():
    """get_dimension caches after first call."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)
    mock_client = AsyncMock()
    # response.json() is called synchronously (no await), so use regular Mock
    mock_response = Mock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 64}]}
    mock_client.post.return_value = mock_response
    provider._client = mock_client

    dim = await provider.get_dimension()
    assert dim == 64
    # Second call should use cache, not call API again
    dim2 = await provider.get_dimension()
    assert dim2 == 64
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_remote_provider_prewarm_success():
    """prewarm creates HTTP client and gets dimension."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)

    mock_client = AsyncMock()
    mock_response = Mock()  # Use Mock, not AsyncMock — response.json() is synchronous
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_client.post.return_value = mock_response

    with patch("memorymesh.embeddings.providers.httpx.AsyncClient", return_value=mock_client):
        await provider.prewarm()
        assert provider._client is not None
        assert provider._dimension == 384


@pytest.mark.asyncio
async def test_remote_provider_prewarm_fallback_dim():
    """prewarm uses default DIM when API call fails."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)

    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("Connection refused")

    with patch("memorymesh.embeddings.providers.httpx.AsyncClient", return_value=mock_client):
        await provider.prewarm()
        assert provider._client is not None
        # Falls back to DIM (384)
        assert provider._dimension == 384


@pytest.mark.asyncio
async def test_remote_provider_close():
    """close cleans up HTTP client."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)
    mock_client = AsyncMock()
    provider._client = mock_client

    await provider.close()
    mock_client.aclose.assert_called_once()
    assert provider._client is None


@pytest.mark.asyncio
async def test_remote_provider_close_no_client():
    """close does nothing when no client."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)
    provider._client = None
    await provider.close()  # Should not raise


@pytest.mark.asyncio
async def test_remote_provider_get_embedding_with_api_key():
    """get_embedding sends Authorization header when api_key set."""
    config = EmbeddingConfig(
        mode="remote",
        remote_api_url="http://localhost:9999",
        remote_api_key="sk-test-key",
    )
    provider = RemoteEmbeddingProvider(config)
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 384}]}
    mock_client.post.return_value = mock_response
    provider._client = mock_client

    emb = await provider.get_embedding("test")
    assert len(emb) == 384
    # Verify auth header was sent
    call_kwargs = mock_client.post.call_args.kwargs
    assert "Authorization" in call_kwargs.get("headers", {})
    assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_remote_provider_get_embedding_api_error():
    """get_embedding raises on API error."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)
    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("API error")
    provider._client = mock_client

    with pytest.raises(Exception, match="API error"):
        await provider.get_embedding("test")


@pytest.mark.asyncio
async def test_remote_provider_get_embedding_fallback_embedding_key():
    """get_embedding falls back to top-level 'embedding' key when 'data' missing."""
    config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
    provider = RemoteEmbeddingProvider(config)
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.json.return_value = {"embedding": [0.5] * 384}
    mock_client.post.return_value = mock_response
    provider._client = mock_client

    emb = await provider.get_embedding("test")
    assert len(emb) == 384
    assert emb[0] == 0.5
