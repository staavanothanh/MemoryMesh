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
  "compact_summary": "string",
  "discussion_topic": "string",
  "work_done": "string",
  "architectural_decisions": "string",
  "last_milestone": "string",
  "next_steps": "string"
}}

Field descriptions:
- narrative_summary: "A single dense paragraph summarizing what was discussed, why it was debated, and the current focus (max 100 words)"
- compact_summary: "A standalone concise paragraph (max 200 words) summarizing key information, decisions, and state — used as session log"
- discussion_topic: "The exact conceptual component or bug being addressed (max 40 words)"
- work_done: "Specific files changed, tests run, commands executed, and concrete achievements this session (max 50 words)"
- architectural_decisions: "Core tech patterns implemented, files changed, or libraries swapped (max 50 words)"
- last_milestone: "The exact functional state at session end — code compiles, tests pass, feature X implemented (max 50 words)"
- next_steps: "Concrete itemized bullet points for the next incoming session (max 60 words)"

Rules:
- narrative_summary is MANDATORY — this is the primary recovery text for cold-start
- compact_summary is MANDATORY — this replaces the separate session compaction step
- Only include information EXPLICITLY present in the log
- If a field has no information, set it to an empty string
- Do not include any conversational filler, intro, or markdown code blocks

Conversation Log:
{log}

Output:"""

COMBINED_AGENT_INSTRUCTION = """[MANDATORY — ALL AGENTS]
Session is auto-managed. new_session() is called automatically.

At conversation start: call recall(query="<topic>") to load past context.

Call commit_milestone(summary, tasks_done, next_steps) ONLY when completing a logical block of work
(e.g., editing multiple files, finishing a feature, fixing a bug, or finalizing a plan).
Do NOT call commit_milestone after every response.

MemoryMesh tracks your uncommitted actions. If you make 5+ actions without committing,
read-only tools (recall, get_session_context) will be blocked until you commit.

If you are in Plan/Read-Only Mode, call commit_milestone ONCE when your final
architectural plan is completed and approved by the user.

[EVERY ~20 EXCHANGES OR BEFORE SHUTDOWN]: call end_session() → continue working

Your text responses are NOT auto-saved. Only tool calls are captured automatically."""

PERMANENT_LOG_DIRECTIVE = """
[PERMANENT LOG DIRECTIVE — ALL AGENTS]:
• Call commit_milestone when finishing a logical block of work.
• Do NOT save after every response — milestone saves are for checkpoints.
• VIOLATION = Uncommitted work may be lost. Call commit_milestone to persist your progress.
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