import pytest
from memorymesh.hooks import HookRegistry


@pytest.fixture
def registry():
    return HookRegistry()


@pytest.mark.asyncio
async def test_register_and_trigger(registry):
    calls = []
    async def my_hook(**kwargs):
        calls.append(kwargs)
    registry.register("after_remember", my_hook)
    await registry.trigger("after_remember", memory_id="abc", user_id="test")
    assert len(calls) == 1
    assert calls[0]["memory_id"] == "abc"


@pytest.mark.asyncio
async def test_multiple_hooks_same_event(registry):
    calls = []
    async def hook1(**kw):
        calls.append("hook1")
    async def hook2(**kw):
        calls.append("hook2")
    registry.register("evt", hook1)
    registry.register("evt", hook2)
    await registry.trigger("evt")
    assert calls == ["hook1", "hook2"]


@pytest.mark.asyncio
async def test_hook_failure_does_not_block(registry):
    calls = []
    async def failing_hook(**kw):
        raise RuntimeError("boom")
    async def good_hook(**kw):
        calls.append("ok")
    registry.register("evt", failing_hook)
    registry.register("evt", good_hook)
    await registry.trigger("evt")
    assert calls == ["ok"]


@pytest.mark.asyncio
async def test_no_hooks_no_error(registry):
    await registry.trigger("nonexistent")
