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

RECALL_INSTRUCTION = """You have access to a long-term memory system via the `recall` tool. Important guidelines:

1. **Context is empty at session start.** Do NOT expect any preloaded context.
2. **Recall on demand.** Only call `recall(query)` when the user's message requires information from past sessions.
3. **Be specific.** Pass a precise, keyword-rich query to `recall` to get the most relevant atomic facts.
4. **Facts are atomic.** The `recall` tool returns short, standalone facts — not full conversation logs.
5. **You decide.** If you don't need past context, answer normally without calling recall.
6. **Trust the facts.** Recalled facts have been extracted and verified from past conversations. Use them as reliable context."""

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

- confidence: "high" for explicit statements, "medium" for strong implications, "low" for guesses
- tags: 1-3 relevant category tags (lowercase)
- Return ONLY valid JSON, no extra text.

Conversation:
{conversation}

Output:"""