---
title: SHL Recommender
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# SHL Conversational Assessment Recommender

A conversational AI agent that helps recruiters select the right SHL assessments through dialogue. It takes vague hiring intents and refines them into grounded shortlists of 1–10 SHL assessment products.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 FastAPI Server                   │
│                  (app/main.py)                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  POST /chat ──► Guardrails ──► Agent ──► Response│
│                (pre-filter)   (LLM)    (validated)│
│                                                  │
│  GET /health ──► {"status": "ok"}                │
│                                                  │
├─────────────────────────────────────────────────┤
│                 Catalog Store                     │
│              (loaded at startup)                  │
│  ┌──────────────┐  ┌───────────────┐            │
│  │ catalog.json  │  │ embeddings.npy│            │
│  │ (377 items)   │  │ (377 × 384)   │            │
│  └──────────────┘  └───────────────┘            │
└─────────────────────────────────────────────────┘
```

### Request Flow

1. **Guardrails** (`app/guardrails.py`) — Fast regex pre-filter catches prompt injection attempts
2. **Retrieval** (`app/catalog.py`) — Semantic search + keyword boost over 377 SHL assessments
3. **Generation** (`app/agent.py`) — Single Gemini API call with system prompt + catalog context + conversation history
4. **Validation** — Post-generation check: every recommended `(name, url)` verified against catalog data
5. **Fallback** — Any error returns a safe, schema-valid response (never a 500)

### Key Design Decisions

- **Stateless**: No per-conversation state stored. Everything re-derived from the `messages` array on every call.
- **Single LLM call**: One Gemini call per `/chat` request. Retrieval is done in code, not via tool-calling.
- **Turn budget**: When `len(messages) >= 5`, the agent commits to best-effort recommendations instead of asking more questions.
- **Anti-hallucination**: Every recommendation is validated against the catalog in code, not just via prompting.
- **Prose naming**: Recommended assessments are always named in the reply text (not just the structured field) so context persists across the stateless API.

## Quick Start

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key

### Setup

```bash
# Clone the repo
git clone <repo-url> && cd shl-recommender

# Create virtual environment (optional but recommended)
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run Locally

```bash
uvicorn app.main:app --port 8000
```

The server will load the catalog + embeddings + embedding model on startup (first run may take 30–60s to download the model; subsequent starts are faster).

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Chat (single turn)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need assessments for a senior Java developer"}]}'
```

## Data Pipeline

The catalog data is already pre-built in `data/`. To regenerate from the raw source:

```bash
# Step 1: Normalize the raw catalog JSON
python3 scripts/build_catalog.py

# Step 2: Compute embeddings
python3 scripts/build_embeddings.py
```

## Testing

```bash
# Run all unit tests
python3 -m pytest tests/ -v

# Run the evaluation harness against a running server
uvicorn app.main:app --port 8000 &
python3 scripts/eval_harness.py --url http://localhost:8000

# Run against deployed URL
python3 scripts/eval_harness.py --url https://your-app.onrender.com
```

## Deployment (Render)

1. Push to GitHub
2. Connect to [Render](https://render.com) and create a new Web Service
3. Select Docker as the runtime
4. Add the `GOOGLE_API_KEY` environment variable
5. Deploy — the `render.yaml` configures everything automatically

The free tier will spin down after inactivity. Cold starts take ~60–90s (model loading).

## Project Structure

```
shl/
├── app/
│   ├── main.py          # FastAPI app, routes, startup
│   ├── schemas.py        # Pydantic models (exact API schema)
│   ├── catalog.py        # Catalog store + semantic search
│   ├── agent.py          # System prompt, LLM call, validation
│   └── guardrails.py     # Injection/off-topic pre-filter
├── data/
│   ├── catalog.json      # Normalized catalog (377 items)
│   └── embeddings.npy    # Precomputed embeddings (377 × 384)
├── scripts/
│   ├── build_catalog.py  # Raw JSON → normalized catalog
│   ├── build_embeddings.py  # Catalog → embeddings
│   └── eval_harness.py   # Replay traces, compute metrics
├── tests/
│   ├── test_schema.py    # Schema compliance tests
│   ├── test_grounding.py # Catalog grounding tests
│   └── test_guardrails.py # Injection/off-topic tests
├── traces/               # 10 sample conversation traces
├── Dockerfile
├── render.yaml
├── requirements.txt
├── .env.example
└── approach.md           # Design document (≤2 pages)
```

## API Reference

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior Java developer"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Add personality tests"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are my updated recommendations including personality: ...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```
