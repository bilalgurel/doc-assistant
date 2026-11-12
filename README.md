**Demo video (2 min):** https://youtu.be/p-JZj0HeIOY

# Document Assistant

An AI assistant that answers a customer's questions using **only their own documents**, shows the source of every answer, and says "That isn't in your documents." instead of guessing.

Built with Python, FastAPI and ChromaDB. Works with Google Gemini or Anthropic Claude; switch with one line in `.env`.

## What it does

- **Ingestion command** loads PDFs, text/markdown files and web pages into a vector store, and re-indexes a document when you run it again.
- **`/chat` endpoint**: question + customer id in, streamed answer + sources out (Server-Sent Events).
- **Per-customer isolation**: every chunk is tagged with a customer id and every search is filtered by it.
- **Cited answers**: each claim carries a reference like `[1]`, shown in the panel as document name + page.
- **Guardrails**:
  - If nothing relevant is found, the fixed refusal message is returned without calling the model.
  - The model is instructed to answer only from the retrieved excerpts and to use the same fixed message otherwise.
  - Token cap per answer and a per-customer rate limit.
- **Question log**: every question, answer, sources used, token counts and cost, stored in SQLite.
- **Admin page** (`/admin`) to edit the system prompt and read the log, protected by a token.
- No API key, model name or price is hard-coded; everything comes from `.env`.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/bilalgurel/doc-assistant.git
cd doc-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Open `.env` and fill in:

- `LLM_PROVIDER`: `gemini` or `anthropic`
- `LLM_MODEL`: the exact model id from your provider's model list
- `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`: the key for the provider you chose
- `ADMIN_TOKEN`: any long random string, used as the admin page password

## Load documents

```bash
python -m app.ingest --customer demo docs/
```

The first run downloads a small embedding model (about 80 MB), so it needs an internet connection once.

You can pass files, folders and URLs together:

```bash
python -m app.ingest --customer demo docs/handbook.pdf https://example.com/faq
```

### Re-index

- Run the same command again: each document is replaced, not duplicated.
- To wipe a customer's index and start clean, add `--reset`:

```bash
python -m app.ingest --customer demo --reset docs/
```

## Run

```bash
uvicorn app.main:app --reload
```

- Chat panel: http://localhost:8000
- Admin page: http://localhost:8000/admin (enter your `ADMIN_TOKEN`, then click Load)

## Try it

With the sample handbook in `docs/` loaded for customer `demo`:

| Question | Expected |
|---|---|
| How much luggage can I bring? | Answer with a source |
| What refund do I get if I cancel 40 days before departure? | Answer with a source |
| When is the remaining balance due? | Answer with a source |
| Do you sell travel insurance? | Answer with a source |
| What is the capital of France? | Refused with the fixed message |
| Do you offer scuba diving lessons? | Refused with the fixed message |

Change the customer field to another id (for example `other`) and ask the same questions: everything is refused, because that customer has no documents.

## API

```
POST /chat
{"customer_id": "demo", "question": "How much luggage can I bring?"}
```

The response is a stream of events:

| Event | Data |
|---|---|
| `token` | `{"text": "..."}` - a piece of the answer |
| `sources` | `{"sources": [{"n": 1, "source": "file.pdf", "page": 3, "excerpt": "..."}]}` |
| `done` | `{"refused": false, "cost_usd": 0.0012}` |
| `error` | `{"message": "..."}` |

## Settings

All optional settings are listed in `.env.example`.

- `MAX_DISTANCE` controls how strict retrieval is. Lower it if out-of-scope questions reach the model; raise it if valid questions are refused too early.
- `MAX_ANSWER_TOKENS` caps each answer. Models that "think" before answering spend part of this budget on thinking, so raise it if answers are cut short.
- `PRICE_INPUT_PER_MTOK` / `PRICE_OUTPUT_PER_MTOK` are only used for the cost column in the log. Set them to your model's current prices.

## Tests

```bash
pytest
```

The tests run offline. They use a small fake embedder and mocked model calls, and cover the Gemini adapter, ingestion, re-indexing, customer isolation, cited answers, refusals, the rate limit and admin access.

## Notes and limits

- This is a demo. In production the customer id should come from your login session, not from the request body.
- The default embedding model works best with English text.
- The rate limiter is in memory, so it resets when the server restarts and is per process.
