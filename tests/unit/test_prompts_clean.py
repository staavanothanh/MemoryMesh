"""Verify instruction prompts have zero forbidden-fruit (Do NOT) patterns."""

from memorymesh.prompts import COMBINED_AGENT_INSTRUCTION, PERMANENT_LOG_DIRECTIVE


def test_combined_agent_instruction_has_no_dont_patterns():
    """COMBINED_AGENT_INSTRUCTION should not contain 'Do NOT' behavioral prohibitions."""
    assert "Do NOT" not in COMBINED_AGENT_INSTRUCTION, (
        "Found forbidden-fruit pattern 'Do NOT' in COMBINED_AGENT_INSTRUCTION"
    )


def test_permanent_log_directive_has_no_dont_patterns():
    """PERMANENT_LOG_DIRECTIVE should not contain 'Do NOT' behavioral prohibitions."""
    assert "Do NOT" not in PERMANENT_LOG_DIRECTIVE, (
        "Found forbidden-fruit pattern 'Do NOT' in PERMANENT_LOG_DIRECTIVE"
    )
