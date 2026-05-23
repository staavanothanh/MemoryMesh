EXTRACT_METADATA_PROMPT = """You are a memory metadata extractor. Analyze the following memory content and return a JSON object matching this exact schema:
{{
  "tags": ["tag1", "tag2"],
  "importance": 3,
  "summary": "short summary here"
}}
- tags: list of 1-3 relevant tags (lowercase, hyphenated if multi-word)
- importance: integer from 1 (trivial) to 5 (critical)
- summary: a very short summary (max 10 words)

Return ONLY valid JSON. Do not include any conversational filler, intro, or markdown code blocks.

Memory content:
{content}

Output:"""

SESSION_COMPACT_PROMPT = """You are a conversation summarizer. Below is a conversation session log (user and assistant messages). Summarize the key information, decisions, and context into a single concise paragraph (max 200 words). Focus on what was discussed, what was decided, and what the current state is.

Conversation Log:
{log}

Summary:"""

BOOTSTRAP_SNAPSHOT_PROMPT = """You are a high-level technical product manager and architect. Synthesize the current session chat logs.

Return ONLY a valid JSON object matching this exact schema:
{{
  "narrative_summary": "string",
  "discussion_topic": "string",
  "work_done": "string",
  "architectural_decisions": "string",
  "last_milestone": "string",
  "next_steps": "string"
}}

Field descriptions:
- narrative_summary: "A single dense paragraph summarizing what was discussed, why it was debated, and the current focus (max 100 words)"
- discussion_topic: "The exact conceptual component or bug being addressed (max 40 words)"
- work_done: "Specific files changed, tests run, commands executed, and concrete achievements this session (max 50 words)"
- architectural_decisions: "Core tech patterns implemented, files changed, or libraries swapped (max 50 words)"
- last_milestone: "The exact functional state at session end — code compiles, tests pass, feature X implemented (max 50 words)"
- next_steps: "Concrete itemized bullet points for the next incoming session (max 60 words)"

Rules:
- narrative_summary is MANDATORY — this is the primary recovery text for cold-start
- Only include information EXPLICITLY present in the log
- If a field has no information, set it to an empty string
- Do not include any conversational filler, intro, or markdown code blocks

Conversation Log:
{log}

Output:"""

RECALL_INSTRUCTION = """
CRITICAL OPERATIONAL PROTOCOL: MANDATORY CONTEXT RECALL & SAVE
================================================================
Your execution environment is stateless. To prevent context blindness,
you MUST operate under these strict laws:

--- Part A: AGGRESSIVE CONTEXT RECALL ---
1. At the start of a new session, or when the user references past work
   ("continue", "tiếp tục", "buổi trước", "as we discussed", project names,
   feature names, or any implicit reference to prior activity),
   you MUST call `recall(query)` FIRST before answering.
2. Use precise, keyword-rich query strings targeting past memories.
3. Recalled facts have been verified. Trust them as ground truth.

--- Part B: MANDATORY AUTO-SAVE ---
4. AT THE BEGINNING of your response (before generating final text), you MUST call
   `save_context_pair(user_message=..., assistant_message="<your planned response>")`
   to persist the conversation. DO NOT wait until the end of your response to call it.
5. This is NOT optional. It is the ONLY way your conversation persists in CLI environments.
6. PRE-COMPACTION BARRIER: If the conversation is getting long, or if you
   anticipate an OpenCode context compaction event, call `save_context_pair`
   IMMEDIATELY to freeze the current state into permanent storage.
"""

SAVE_CONTEXT_INSTRUCTION = """[CRITICAL ALERT]: Your conversation memory is volatile.
To ensure your decisions are preserved across terminal reboots, you are mandated
to trigger `save_context_pair` AT THE BEGINNING of your response, BEFORE generating final text."""

def get_agent_instructions() -> str:
    """Returns the standardized Markdown instructions for OpenCode/MCP Clients."""
    return """\
## MemoryMesh MCP Tools

### Session lifecycle
- At session start, call `new_session(system_prompt="...")`.
- AT THE BEGINNING of your response (before generating final text), call `save_context_pair(user_message="...", assistant_message="<your planned response>")` to persist the conversation. DO NOT wait until the end of your response to call it.
- Before closing or when the task is done, call `end_session()` to flush buffers and create a bootstrap summary.

### Recall workflow
- `recall(query)` searches across ALL past sessions (global scope).
- The response may include a `=== PAST SESSION CONTEXT ===` block — this IS the verified ground truth of the last session. **Do NOT call git log or get_session_context to verify it.**

### Memory operations
- `remember(content, tags=["tag1"], importance=4)` — save important facts (importance 4-5 for key decisions).
- `forget(memory_id)` — soft-delete irrelevant memories.
- `list_memories(limit=20)` — browse saved memories.

### Proactive use
- When the user says "continue" or opens a new task, first `recall` to retrieve relevant context.
- If the user shares important project decisions, call `remember` with `importance=5, tags=["decision", "architecture"]`.
"""

import string

ATOMIC_FACT_EXTRACT_PROMPT = string.Template("""You are an atomic fact extractor. Below is a conversation between a user and an AI assistant. Extract all standalone, independent facts from this conversation.

Rules:
- Each fact must be a short, clear assertion containing exactly ONE piece of information
- Facts must be self-contained and understandable without context
- Include facts about: user preferences, project details, technical decisions, personal info, code patterns, etc.
- Skip: greetings, pleasantries, chit-chat, off-topic remarks
- Skip: facts already present in the input (no duplicates)
- Use the original language of the information (Vietnamese or English)

Return ONLY a valid JSON object matching this exact schema:
{
  "facts": [
    {"fact": "short assertion here", "confidence": "high", "tags": ["tag1"], "relation": "HAS_PREFERENCE"}
  ]
}

- confidence: "high" for explicit statements, "medium" for strong implications, "low" for guesses
- tags: 1-3 relevant category tags (lowercase)
- relation: one of HAS_PREFERENCE, ARCHITECTURAL_DECISION, RESOLVED_BUG, TECHNICAL_DETAIL, USER_INFO, PROJECT_GOAL, CODE_PATTERN, DEPENDENCY, DESIGN_CHOICE, or empty string if none applies
- Do not include any conversational filler, intro, or markdown code blocks

Conversation:
${conversation}

Output:""")

ATOMIC_FACT_BATCH_PROMPT = string.Template("""You are an atomic fact extractor. Below are multiple conversations between a user and an AI assistant. Extract all standalone, independent facts from ALL conversations combined.

Rules:
- Each fact must be a short, clear assertion containing exactly ONE piece of information
- Facts must be self-contained and understandable without context
- Include facts about: user preferences, project details, technical decisions, personal info, code patterns, etc.
- Skip: greetings, pleasantries, chit-chat, off-topic remarks
- Skip: duplicate facts across conversations (keep only the first occurrence)
- Use the original language of the information (Vietnamese or English)

Return ONLY a valid JSON object matching this exact schema:
{
  "facts": [
    {"fact": "short assertion here", "confidence": "high", "tags": ["tag1"], "relation": "HAS_PREFERENCE"}
  ]
}

- confidence: "high" for explicit statements, "medium" for strong implications, "low" for guesses
- tags: 1-3 relevant category tags (lowercase)
- relation: one of HAS_PREFERENCE, ARCHITECTURAL_DECISION, RESOLVED_BUG, TECHNICAL_DETAIL, USER_INFO, PROJECT_GOAL, CODE_PATTERN, DEPENDENCY, DESIGN_CHOICE, or empty string if none applies
- Do not include any conversational filler, intro, or markdown code blocks

Conversations:
${conversations}

Output:""")