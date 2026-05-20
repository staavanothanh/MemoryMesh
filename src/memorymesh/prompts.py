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

ATOMIC_FACT_EXTRACT_PROMPT = """You are an atomic fact extractor. Below is a conversation between a user and an AI assistant. Extract all standalone, independent facts from this conversation.

Rules:
- Each fact must be a short, clear assertion containing exactly ONE piece of information
- Facts must be self-contained and understandable without context
- Include facts about: user preferences, project details, technical decisions, personal info, code patterns, etc.
- Skip: greetings, pleasantries, chit-chat, off-topic remarks
- Skip: facts already present in the input (no duplicates)
- Use the original language of the information (Vietnamese or English)

Return a JSON object with a "facts" field containing an array of fact objects:
{
  "facts": [
    {"fact": "short assertion here", "confidence": "high|medium|low", "tags": ["tag1", "tag2"]},
    {"fact": "another assertion", "confidence": "high", "tags": ["tag1"]}
  ]
}

- confidence: "high" for explicit statements, "medium" for strong implications, "low" for猜测
- tags: 1-3 relevant category tags (lowercase)
- Return ONLY valid JSON, no extra text.

Conversation:
{conversation}

Output:"""