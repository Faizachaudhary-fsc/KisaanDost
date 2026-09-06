# KisaanDost Backend

Python + FastAPI backend for KisaanDost: a voice-first Urdu agricultural
assistant and crop marketplace for Pakistani farmers.

The mobile frontend in `frontend/` is owned separately. Its API contract in
[`frontend/services/api.ts`](../frontend/services/api.ts) is the **source of
truth** for every endpoint and response shape here — this backend conforms to
it and must not change it.

## Status

The backend foundation is complete and runnable. The AI pipeline is **not**
implemented yet: the four service modules are placeholders returning
deterministic stub data, so all endpoints are callable end-to-end today.

`GET /health` reports which stages are still stubbed.

## Quick start

```bash
cd backend

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env       # optional — defaults work without it

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: <http://localhost:8000/docs>

### No MongoDB? Still works.

Leave `MONGODB_URI` empty and the listings API falls back to an in-memory
store, so the whole marketplace flow is testable before credentials exist.
That store is per-process and **not durable** — data is lost on restart.
`/health` shows which backing store is live.

### Connecting the app

Use your machine's LAN IP, not `localhost` — an emulator or physical device
resolves `localhost` to itself:

```ts
// frontend/services/api.ts
const USE_MOCK = false;
const BASE_URL = 'http://192.168.x.x:8000';
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/assistant/voice` | Voice question → transcription, answer, TTS audio |
| `POST` | `/api/listings` | Create a crop listing |
| `GET` | `/api/listings` | List listings; optional `crop`, `location` filters |
| `GET` | `/api/listings/{id}` | Fetch one listing |
| `DELETE` | `/api/listings/{id}` | Delete one listing |
| `GET` | `/health` | Storage + pipeline status (not in the contract) |

### Response envelope

Success bodies carry `success: true` plus a payload key (`listing`,
`listings`, or the voice fields). **Every** failure — including validation
errors and 404s — returns:

```json
{ "success": false, "error": "Listing not found" }
```

FastAPI's default validation response (HTTP 422, `{"detail": [...]}`) does not
match the contract, so `app/main.py` installs exception handlers that rewrite
all error paths into the envelope above. Removing them silently breaks the
app's error UI, which reads `result.error`.

## Layout

```
backend/
├── app/
│   ├── main.py                     FastAPI entry point, CORS, error handlers
│   ├── config.py                   Settings from .env, safe defaults
│   ├── database.py                 MongoDB connection (optional, non-fatal)
│   ├── models/                     Pydantic request/response schemas
│   │   ├── common.py               Error + health envelopes
│   │   ├── listing.py              Listing schemas and the Crop enum
│   │   └── assistant.py            Voice response schema
│   ├── routes/
│   │   ├── assistant.py            POST /api/assistant/voice
│   │   └── listings.py             CRUD for /api/listings
│   ├── repositories/
│   │   └── listing_repository.py   MongoDB access + in-memory fallback
│   └── services/                   AI pipeline stages (all placeholders)
│       ├── speech_to_text.py       Urdu ASR
│       ├── rag.py                  Agricultural retrieval
│       ├── llm.py                  Answer generation
│       └── text_to_speech.py       Urdu TTS
├── tests/test_api_contract.py      Contract conformance tests
├── requirements.txt
├── .env.example
└── .gitignore
```

## Tests

```bash
cd backend && pytest
```

These assert response *shapes*, not stub content, so they stay valid once the
real AI pipeline lands. A failure there means the contract broke.

## Next: the AI pipeline

Replace the four stubs in `app/services/`. Each has a `TODO(ai-pipeline)`
marker and a fixed signature, so routes and models need no changes:

```
audio → speech_to_text → rag → llm → text_to_speech → audio_base64
```

Flip each module's `IMPLEMENTED` flag to `True` as it is completed; `/health`
picks that up automatically.
