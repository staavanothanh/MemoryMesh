import json
import pytest
from memorymesh.utils.json_parser import clean_and_parse_llm_json


class TestCleanAndParseLlmJson:

    def test_clean_json(self):
        result = clean_and_parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_markdown_fence(self):
        result = clean_and_parse_llm_json("""```json
{"name": "test", "count": 3}
```""")
        assert result == {"name": "test", "count": 3}

    def test_json_with_bare_fence(self):
        result = clean_and_parse_llm_json("""```
{"key": 42}
```""")
        assert result == {"key": 42}

    def test_json_with_output_prefix(self):
        result = clean_and_parse_llm_json("""Output: {"valid": true}""")
        assert result == {"valid": True}

    def test_json_with_prefix_and_fence(self):
        result = clean_and_parse_llm_json("""Output:
```json
{"msg": "hello"}
```""")
        assert result == {"msg": "hello"}

    def test_json_with_surrounding_text(self):
        raw = """Here is the result:
{"key": "value"}
Hope this helps!"""
        result = clean_and_parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_json_with_truncated_curly(self):
        raw = '{"project": "MemoryMesh", "status": "active"'
        with pytest.raises(json.JSONDecodeError):
            clean_and_parse_llm_json(raw)

    def test_json_array(self):
        result = clean_and_parse_llm_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_nested_json(self):
        raw = '{"a": {"b": [1, 2, 3]}, "c": null}'
        result = clean_and_parse_llm_json(raw)
        assert result == {"a": {"b": [1, 2, 3]}, "c": None}

    def test_empty_string(self):
        with pytest.raises(json.JSONDecodeError):
            clean_and_parse_llm_json("")

    def test_non_json_text(self):
        with pytest.raises(json.JSONDecodeError):
            clean_and_parse_llm_json("Just plain text without JSON")

    def test_partial_json_recovery(self):
        raw = 'Some text before {"found": true} and some after'
        result = clean_and_parse_llm_json(raw)
        assert result == {"found": True}

    def test_fallback_obj_extraction(self):
        raw = "Here is your data: {\"status\": \"ok\", \"count\": 5} End."
        result = clean_and_parse_llm_json(raw)
        assert result == {"status": "ok", "count": 5}

    def test_fallback_array_extraction(self):
        result = clean_and_parse_llm_json("Results: [{\"id\": 1}, {\"id\": 2}]")
        assert result == {"id": 1}
