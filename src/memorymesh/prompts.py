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

BOOTSTRAP_SNAPSHOT_PROMPT = """You are a high-level technical product manager and architect. Synthesize the current session chat logs.

Return a strict JSON object with these fields (in this order):
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
- Return ONLY valid JSON, no extra text

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
4. After EVERY user+assistant exchange, you MUST call
   `save_context_pair(user_message=..., assistant_message=...)`
   with the EXACT raw text exchanged. Do NOT summarize or omit.
5. This is NOT optional. It is the ONLY way your conversation persists.
6. PRE-COMPACTION BARRIER: If the conversation is getting long, or if you
   anticipate an OpenCode context compaction event, call `save_context_pair`
   IMMEDIATELY to freeze the current state into permanent storage.
"""

SAVE_CONTEXT_INSTRUCTION = """[CRITICAL ALERT]: Your conversation memory is volatile.
To ensure your decisions are preserved across terminal reboots, you are mandated
to trigger `save_context_pair` immediately after your text stream resolution."""

ATOMIC_FACT_EXTRACT_PROMPT = """You are an atomic fact extractor. Below is a conversation between a user and an AI assistant. Extract all standalone, independent facts from this conversation.

Rules:
- Each fact must be a short, clear assertion containing exactly ONE piece of information
- Facts must be self-contained and understandable without context
- Include facts about: user preferences, project details, technical decisions, personal info, code patterns, etc.
- Skip: greetings, pleasantries, chit-chat, off-topic remarks
- Skip: facts already present in the input (no duplicates)
- Use the original language of the information (Vietnamese or English)

Return a JSON object with a "facts" field containing an array of fact objects:
{{
  "facts": [
    {{"fact": "short assertion here", "confidence": "high|medium|low", "tags": ["tag1", "tag2"], "relation": "HAS_PREFERENCE"}},
    {{"fact": "another assertion", "confidence": "high", "tags": ["tag1"], "relation": "ARCHITECTURAL_DECISION"}}
  ]
}}

- confidence: "high" for explicit statements, "medium" for strong implications, "low" for guesses
- tags: 1-3 relevant category tags (lowercase)
- relation: one of HAS_PREFERENCE, ARCHITECTURAL_DECISION, RESOLVED_BUG, TECHNICAL_DETAIL, USER_INFO, PROJECT_GOAL, CODE_PATTERN, DEPENDENCY, DESIGN_CHOICE, or empty string if none applies
- Return ONLY valid JSON, no extra text.

Conversation:
{conversation}

Output:"""

ATOMIC_FACT_BATCH_PROMPT = """You are an atomic fact extractor. Below are multiple conversations between a user and an AI assistant. Extract all standalone, independent facts from ALL conversations combined.

Rules:
- Each fact must be a short, clear assertion containing exactly ONE piece of information
- Facts must be self-contained and understandable without context
- Include facts about: user preferences, project details, technical decisions, personal info, code patterns, etc.
- Skip: greetings, pleasantries, chit-chat, off-topic remarks
- Skip: duplicate facts across conversations (keep only the first occurrence)
- Use the original language of the information (Vietnamese or English)

Return a JSON object with a "facts" field containing an array of fact objects:
{{
  "facts": [
    {{"fact": "short assertion here", "confidence": "high|medium|low", "tags": ["tag1", "tag2"], "relation": "HAS_PREFERENCE"}},
    {{"fact": "another assertion", "confidence": "high", "tags": ["tag1"], "relation": "ARCHITECTURAL_DECISION"}}
  ]
}}

- confidence: "high" for explicit statements, "medium" for strong implications, "low" for guesses
- tags: 1-3 relevant category tags (lowercase)
- relation: one of HAS_PREFERENCE, ARCHITECTURAL_DECISION, RESOLVED_BUG, TECHNICAL_DETAIL, USER_INFO, PROJECT_GOAL, CODE_PATTERN, DEPENDENCY, DESIGN_CHOICE, or empty string if none applies
- Return ONLY valid JSON, no extra text.

Conversations:
{conversations}

Output:"""