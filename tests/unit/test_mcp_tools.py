import pytest
from memorymesh.mcp_server.tools import _schema, TOOLS, MUTATE_TOOLS, COGNITIVE_TOOLS, EXPENSIVE_TOOLS
from memorymesh.schemas import PingInput

def test_schema_stripping():
    schema = _schema(PingInput)
    assert "title" not in schema
    # PingInput doesn't have required properties by default maybe, but if it has user_id it shouldn't have title
    if "properties" in schema:
        for prop in schema["properties"].values():
            assert "title" not in prop

def test_tools_list_valid():
    assert len(TOOLS) > 0
    for tool in TOOLS:
        assert tool.name
        assert tool.description
        assert isinstance(tool.inputSchema, dict)

def test_tool_groups():
    tool_names = {t.name for t in TOOLS}
    for t in MUTATE_TOOLS:
        assert t in tool_names
    for t in COGNITIVE_TOOLS:
        assert t in tool_names
    for t in EXPENSIVE_TOOLS:
        assert t in tool_names