import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from memorymesh.router import RouterClient
from memorymesh.config import RouterConfig
from memorymesh.errors import RouterError, LLMUnavailableError


@pytest.fixture
def config():
    return RouterConfig(
        url="http://mock-router:20128/v1",
        default_model="test-model",
        fallback_model="test-fallback",
        timeout_s=5,
        max_retries=2,
    )


@pytest.fixture
def client(config):
    return RouterClient(config)


@pytest.mark.asyncio
async def test_call_llm_success(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"choices":[{"message":{"content":"Hello"}}]}'
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "Hello"}}]})

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        result = await client.call_llm("ping")
        assert result == "Hello"
        assert client._failure_count == 0


@pytest.mark.asyncio
async def test_call_llm_retry_then_fallback(client):
    responses = [
        MagicMock(status_code=500, text="Server Error"),
        MagicMock(status_code=500, text="Server Error"),
        MagicMock(status_code=200, text='{"choices":[{"message":{"content":"Fallback OK"}}]}',
                  json=MagicMock(return_value={"choices": [{"message": {"content": "Fallback OK"}}]})),
    ]
    mock_post = AsyncMock(side_effect=responses)

    with patch("httpx.AsyncClient.post", new=mock_post):
        result = await client.call_llm("ping")
        assert result == "Fallback OK"


@pytest.mark.asyncio
async def test_call_llm_all_fail_raises_router_error(client):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Server Error"

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(RouterError):
            await client.call_llm("ping")


@pytest.mark.asyncio
async def test_circuit_breaker_after_three_failures(client):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Server Error"

    client._failure_count = 2
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(LLMUnavailableError):
            await client.call_llm("ping")


@pytest.mark.asyncio
async def test_parse_json_from_raw_decode(client):
    import json as json_lib
    raw = '{"choices":[{"message":{"content":"Parsed"}}]} some trailing text'
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = raw
    mock_response.json = MagicMock(side_effect=json_lib.JSONDecodeError("Invalid JSON", raw, 0))

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        result = await client.call_llm("ping")
        assert result == "Parsed"


@pytest.mark.asyncio
async def test_timeout_raises_router_error(client):
    import httpx
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
        with pytest.raises(RouterError):
            await client.call_llm("ping")
