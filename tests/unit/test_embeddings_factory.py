import pytest
import sys
from unittest.mock import patch, MagicMock
from memorymesh.config import EmbeddingConfig
from memorymesh.embeddings.factory import create_embedding_provider, has_local_embedding
from memorymesh.embeddings.providers import RemoteEmbeddingProvider, LocalEmbeddingProvider

def test_has_local_embedding_true():
    with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
        assert has_local_embedding() is True

def test_has_local_embedding_false():
    with patch.dict('sys.modules', {'sentence_transformers': None}):
        assert has_local_embedding() is False

def test_create_embedding_provider_remote():
    config = EmbeddingConfig(mode="remote", remote_api_url="http://test")
    provider = create_embedding_provider(config)
    assert isinstance(provider, RemoteEmbeddingProvider)

def test_create_embedding_provider_local():
    config = EmbeddingConfig(mode="local", model="test-model")
    with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
        with patch('memorymesh.embeddings.providers.LocalEmbeddingProvider.__init__', return_value=None) as mock_init:
            provider = create_embedding_provider(config)
            assert isinstance(provider, LocalEmbeddingProvider)
            mock_init.assert_called_once_with("test-model")

def test_create_embedding_provider_local_fallback():
    # If mode is remote but no url is provided, it falls back to local
    config = EmbeddingConfig(mode="remote", remote_api_url="", model="test-model")
    with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
        with patch('memorymesh.embeddings.providers.LocalEmbeddingProvider.__init__', return_value=None) as mock_init:
            provider = create_embedding_provider(config)
            assert isinstance(provider, LocalEmbeddingProvider)
            mock_init.assert_called_once_with("test-model")

def test_create_embedding_provider_local_missing_sentence_transformers_local():
    config = EmbeddingConfig(mode="local", model="test-model")
    with patch.dict('sys.modules', {'sentence_transformers': None}):
        with pytest.raises(RuntimeError, match="Local embedding requires sentence-transformers"):
            create_embedding_provider(config)

def test_create_embedding_provider_local_missing_sentence_transformers_remote_fallback():
    config = EmbeddingConfig(mode="remote", remote_api_url="", model="test-model")
    with patch.dict('sys.modules', {'sentence_transformers': None}):
        with pytest.raises(RuntimeError, match="Remote embedding API not configured and local embedding not available"):
            create_embedding_provider(config)