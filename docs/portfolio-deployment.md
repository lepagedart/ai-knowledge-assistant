# Portfolio deployment

This is a small reference deployment for a portfolio demonstration, not an
enterprise production service. It has no accounts, authentication, durable
sessions, persistent client storage, production SLA, OCR, or multi-tenant
isolation.

## Configuration

Set `AI_KNOWLEDGE_ASSISTANT_ENV=production`, `FLASK_SECRET_KEY`, and
`OPENAI_API_KEY` in the host's secret manager. Production startup rejects a
missing secret key with a stable message that does not include values. The WSGI
application intentionally does not require or contact OpenAI at startup; a
missing API key is instead reported through the existing sanitized provider
failure when an indexing or answer operation needs it. `OPENAI_EMBEDDING_MODEL`
and `OPENAI_ANSWER_MODEL` are optional model overrides.

Optional deployment controls are `WORKSPACE_ROOT`, `KNOWLEDGE_RUN_TTL_SECONDS`
(default `3600`), `ALLOW_CLIENT_UPLOADS` (default `true`), `MAX_REQUEST_BYTES`
(default `53477376`, including multipart overhead), `MAX_FILE_BYTES` (default `10485760`),
`MAX_DOCUMENTS_PER_RUN` (default `12`), `MAX_TOTAL_UPLOAD_BYTES` (default
`52428800`), `MAX_CHUNKS_PER_RUN` (default `2000`), `MAX_QUESTION_LENGTH`
(default `2000`), and `MAX_ACTIVE_RUNS` (default `25`). Existing structured
CSV/XLSX row and cell bounds remain enforced. A rejected limit is reported;
source content is never silently truncated. Keep `MAX_REQUEST_BYTES` at least
large enough for the selected aggregate-upload limit and multipart overhead.

`PORT` is supplied by Render and is consumed by Gunicorn in the start command;
local direct development defaults to `5000`. It is not hardcoded for deployment.

For a public portfolio demo, set `ALLOW_CLIENT_UPLOADS=false`. The synthetic
Harbor & Hearth demo remains available, while the route also rejects uploads.
This avoids inviting confidential uploads and reduces privacy/support exposure.
Private, client-controlled deployments can explicitly enable uploads.

## Run state and lifecycle

Each browser has an opaque random run ID in its signed Flask cookie. Documents,
indexes, answers, and reconciliation state stay in a unique temporary server
workspace and process memory. A reset removes only the current browser's run.
The registry is lock-protected for normal concurrent requests in one process;
it is not durable or a production multi-tenant architecture.

Expired runs are pruned on normal application requests using server-side monotonic time.
Their workspaces are safely removed after the configured TTL. Abandoned runs
can remain until another application request arrives or the process restarts.
Health probes deliberately do not trigger cleanup.

Run state is process-local, so deploy **exactly one Gunicorn worker and one
thread**. Do not horizontally scale or increase workers without replacing this
state architecture with durable session routing/storage. This deployment does
not add Redis, a database, or a queue.

## Render

The committed `render.yaml` pins Python `3.12.1`, uses `pip install .`,
`/health`, and one Gunicorn worker/thread. Create a Render Blueprint or web
service from the repository, then confirm `FLASK_SECRET_KEY` and
`OPENAI_API_KEY` are configured as secrets. Render terminates HTTPS;
Flask/Gunicorn do not manage certificates. The app does not generate external
absolute URLs, so no forwarded-header trust middleware is needed; it deliberately
does not trust arbitrary proxy headers.

Production command:

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 ai_knowledge_assistant.wsgi:app
```

`GET /health` returns only `{"status":"ok"}` and makes no provider, document,
or filesystem call. Production cookies are Secure, HttpOnly, and SameSite=Lax;
development keeps Secure off for local HTTP. Forms have session-bound CSRF
tokens. Flask debug is forced off.

OpenAI calls remain bounded: embeddings use their existing bounded batches, an
active run indexes once, answer context remains bounded, and a supported
question has at most one response request with no retries. A public demo
incurs API usage and should be monitored. No cross-user upload/index cache is
used.
