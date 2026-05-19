EXTRACT_METADATA_PROMPT = """You are a memory metadata extractor. Analyze the following memory content and return a JSON object with these fields:
- tags: list of 1-3 relevant tags (lowercase, hyphenated if multi-word)
- importance: integer from 1 (trivial) to 5 (critical)
- summary: a very short summary (max 10 words)

Return ONLY valid JSON, nothing else.

Memory content:
{content}

Output:"""