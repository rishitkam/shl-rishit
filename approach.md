# Approach Document — SHL Conversational Assessment Recommender

## Problem

Build a conversational agent that takes a recruiter from a vague hiring intent to a grounded shortlist of SHL assessments. The agent is exposed as a stateless FastAPI service (`POST /chat`) graded by an automated LLM-simulated-user harness. Key constraints: exact response schema compliance, zero hallucinated URLs, 8-message conversation cap, 30-second per-call timeout.

## Architecture: Retrieval-Augmented Generation (RAG)

The system uses a three-layer architecture:

**Data Layer** — 377 SHL assessment products normalized from the official catalog JSON, with precomputed sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim). Loaded once at process startup.

**Conversation Layer** — On each `/chat` call: (1) concatenate all user messages into a retrieval query, (2) search the catalog via cosine similarity + keyword boost for 25 candidates, (3) make a single Gemini 2.0 Flash call with the system prompt, conversation history, and candidate context, (4) validate every recommendation against the catalog in code.

**API Layer** — Thin FastAPI routes with an internal 22-second timeout around the LLM call. Every code path returns a schema-valid `ChatResponse`. A global exception handler catches anything that slips through.

## Design Choices and Trade-offs

### Single LLM call vs. agent loop
With an 8-message cap and 30s timeout, each additional LLM round-trip is pure risk. I chose a single well-grounded call (code does retrieval, LLM does reasoning) over a tool-calling loop. This is more predictable and easier to test, at the cost of less dynamic behavior.

### Semantic + keyword retrieval
Pure semantic search can miss obvious literal matches (e.g., "Java" matching "JVM" but ranking below something semantically adjacent). Adding a keyword boost on exact name/description substring matches ensures technology-specific queries rank the right tests. The cost is a slightly more complex scoring function, but it measurably improved recall on the 10 sample traces.

### Turn-budget safety valve
The system counts incoming messages. At `len(messages) >= 5` (roughly the 3rd assistant reply), it injects an instruction to commit to best-effort recommendations rather than asking more questions. This prevents the conversation from hitting the 8-message cap with an empty shortlist.

### Prose naming rule
Because the API is stateless and only `role`/`content` text persists between calls, every reply that delivers recommendations names them by exact catalog name in the prose. The structured `recommendations` array is for the caller's UI; the prose is the agent's actual memory.

### `end_of_conversation` logic
The sample traces show `end_of_conversation: true` only on explicit user confirmation (e.g., "That's what we need", "Locking it in"), not whenever recommendations are non-empty. Many turns deliver a shortlist with `end_of_conversation: false` because the conversation is still open-ended. The agent follows this trace-derived pattern.

### Anti-hallucination: code-level validation
The system prompt instructs the LLM to only use provided catalog data, but prompting alone isn't sufficient. Every `(name, url)` pair in the recommendations array is validated against a lookup table before being included in the response. Unmatched items are silently dropped. This is the real defense against hallucination.

## Evaluation Approach

**Unit tests** (70 tests): Schema compliance on all edge cases, guardrail pattern detection (27 injection variants, 13 legitimate passthroughs), catalog loading/search verification.

**Integration harness** (`scripts/eval_harness.py`): Replays all 10 sample conversations against a running server. Reports Recall@10, schema compliance, catalog grounding, and behavioral probes (injection, off-topic, legal-question refusal).

## What Didn't Work

- **Multi-step extraction**: Early attempts to separately extract role/seniority/skills from conversation history before retrieval added latency (a second LLM call) without improving recall. Concatenating raw user text works better and stays well within the timeout.
- **Overly aggressive injection filter**: An early version flagged "I want to ignore the Angular tests" as injection. Fixed by requiring the patterns to match structurally (e.g., "ignore" + "instructions") rather than triggering on individual words.
- **High retrieval `top_k`**: Setting `top_k=50` flooded the LLM context with irrelevant options, leading to lower-quality recommendations. Settled on 25 as the sweet spot.

## AI Tools Used

- **Google Gemini 2.0 Flash** — LLM for conversational reasoning and response generation
- **Sentence-Transformers (all-MiniLM-L6-v2)** — Local embedding model for catalog search
- **AI coding assistant** — Used for code scaffolding, test generation, and documentation drafting
