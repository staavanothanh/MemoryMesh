"""Test uncovered paths in embedder.py: prewarm_embedder, get_embedding_dimension auto-init, close_embedder."""

import pytest
from unittest.mock import patch, AsyncMock

from memorymesh.embedder import (
    prewarm_embedder,
    get_embedding_dimension,
    close_embedder,
)
from memorymesh.embeddings.providers import EmbeddingProvider


@pytest.mark.asyncio
async def test_prewarm_embedder_already_initialized():
    """prewarm_embedder returns early if _provider is already set."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    with patch("memorymesh.embedder._provider", mock_provider):
        with patch("memorymesh.embedder.init_embedder", new=AsyncMock()) as m_init:
            await prewarm_embedder("some-model")
            m_init.assert_not_called()


@pytest.mark.asyncio
async def test_prewarm_embedder_auto_init():
    """prewarm_embedder auto-initializes when _provider is None."""
    with patch("memorymesh.embedder._provider", None):
        with patch("memorymesh.embedder.init_embedder", new=AsyncMock()) as m_init:
            await prewarm_embedder("test-model")
            m_init.assert_called_once()
            args = m_init.call_args[0][0]
            assert args.mode == "local"
            assert args.model == "test-model"


@pytest.mark.asyncio
async def test_get_embedding_dimension_auto_init():
    """get_embedding_dimension auto-initializes when _provider is None."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    mock_provider.get_dimension.return_value = 384

    with patch("memorymesh.embedder._provider", mock_provider):
        with patch("memorymesh.embedder.init_embedder", new=AsyncMock()) as m_init:
            dim = await get_embedding_dimension()
            assert dim == 384
            # _provider is already set, so init_embedder should not be called
            m_init.assert_not_called()


@pytest.mark.asyncio
async def test_get_embedding_dimension_with_none_provider():
    """get_embedding_dimension auto-inits when _provider is None then returns dim."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    mock_provider.get_dimension.return_value = 128

    with patch("memorymesh.embedder._provider", None):
        with patch("memorymesh.embedder.init_embedder", new=AsyncMock()) as m_init:
            with patch("memorymesh.embedder._provider", mock_provider):
                dim = await get_embedding_dimension()
                assert dim == 128


@pytest.mark.asyncio
async def test_get_embedding_dimension_returns_from_provider():
    """get_embedding_dimension returns dimension from existing provider."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    mock_provider.get_dimension.return_value = 128
    with patch("memorymesh.embedder._provider", mock_provider):
        dim = await get_embedding_dimension()
        assert dim == 128
        mock_provider.get_dimension.assert_called_once()


@pytest.mark.asyncio
async def test_close_embedder_closes_provider():
    """close_embedder calls close on provider and sets it to None."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    with patch("memorymesh.embedder._provider", mock_provider):
        await close_embedder()
        mock_provider.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_embedder_no_provider():
    """close_embedder does nothing when _provider is None."""
    with patch("memorymesh.embedder._provider", None):
        await close_embedder()  # Should not raise


@pytest.mark.asyncio
async def test_prewarm_embedder_with_none_provider():
    """prewarm_embedder auto-initializes via init_embedder."""
    with patch("memorymesh.embedder._provider", None):
        with patch("memorymesh.embedder.init_embedder", new=AsyncMock()) as m_init:
            await prewarm_embedder("test-model")
            m_init.assert_called_once()
