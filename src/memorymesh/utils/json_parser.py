import re
import json
from typing import Any


def clean_and_parse_llm_json(response_text: str) -> Any:
    """Strip markdown fences and parse LLM JSON response.

    Handles:
    - ```json ... ``` fences
    - ``` ... ``` fences
    - "Output:" or "Output:" prefix
    - Leading/trailing whitespace
    - Truncated JSON (best-effort recovery)
    """
    text = response_text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Strip common prefixes
    text = re.sub(r"^(?:Output|Output):\s*", "", text)

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON object/array in text
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
        if start >= 0:
            candidate = text[start:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError(
        f"Could not parse LLM response as JSON: {response_text[:500]}",
        response_text,
        0,
    )
