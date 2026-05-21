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

BOOTSTRAP_SNAPSHOT_PROMPT = """You are a workspace state condenser. Below is a conversation session log from a software development session. 

Extract a structured summary of ONLY what is explicitly discussed. Return a JSON object with these fields:
- project_identity: what project/repo was being worked on (max 20 words)
- discussion_topic: the main topic or narrative of this session (max 50 words)
- architectural_decisions: key architecture/design decisions made (max 50 words)
- last_milestone: what was accomplished this session (max 50 words)
- open_impediments: unresolved issues or blockers mentioned (max 50 words)
- next_steps: planned next actions or priorities (max 50 words)

Rules:
- Only include information EXPLICITLY present in the log
- If a field has no information, set it to an empty string
- Be concise: each field max 50 words
- Return ONLY valid JSON, no extra text

Conversation Log:
{log}

Output:"""

RECALL_INSTRUCTION = """CRITICAL: You MUST use the `recall` tool to retrieve past context.

RULES:
1. When the user references past work — "continue", "as we discussed", "the bug we fixed",
   "my preferences", "last session", project names, feature names, or any implicit reference
   to prior activity — you MUST call `recall(query)` BEFORE answering.
2. Context starts empty at session start. Do NOT rely on your training data for project context.
3. Use precise, keyword-rich queries: include the user's exact words plus related terms.
4. Recalled facts have been verified. Trust them as ground truth.
5. Examples of triggers: "let's continue", "like before", "hôm trước", "cái bug đó",
   any question about user preferences, project architecture, or past decisions.
6. ONLY skip recall for purely general knowledge or simple greetings."""

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