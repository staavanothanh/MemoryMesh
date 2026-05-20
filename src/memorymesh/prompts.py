EXTRACT_METADATA_PROMPT = """You are a memory metadata extractor. Analyze the following memory content and return a JSON object with these fields:
- tags: list of 1-3 relevant tags (lowercase, hyphenated if multi-word)
- importance: integer from 1 (trivial) to 5 (critical)
- summary: a very short summary (max 10 words)

Return ONLY valid JSON, nothing else.

Memory content:
{content}

Output:"""

SESSION_COMPACT_PROMPT = """You are a conversation summarizer. Below is a conversation session log (user and assistant messages). Summarize the key information, decisions, and context into a single concise paragraph (max 200 words). Focus on what was discussed, what was decided, and what the current state is.

Conversation Log:
{log}

Summary:"""