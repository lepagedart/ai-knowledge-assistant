# AI Knowledge Assistant

A small, grounded business knowledge assistant reference implementation for the
**Custom AI Knowledge Assistant — starting at $550** service. It demonstrates a
simple workflow: a business provides internal reference materials, the
application prepares them for search, and team members ask questions answered
from those materials.

This is a public portfolio/reference repository. It contains only fictional,
synthetic demo content. Never add real client documents, API keys, or local
runtime data to this repository.

## V1 outcome

### Structured business records (V1)

Alongside ordinary knowledge documents, V1 accepts UTF-8 CSV and non-macro XLSX
business records. Explicit aliases are normalized deterministically and each
invoice, purchase-order, vendor, product, or generic row becomes citation-ready
evidence. It does not perform calculations, reconciliation, recommendations,
OCR, or automatic actions. Harbor & Hearth fixtures are fictional.

The finished V1 is intended to support one business and one knowledge
assistant. It will accept a reasonably sized collection of standard business
documents in these formats:

- PDF
- DOCX
- TXT
- Markdown

V1 applies a 10 MiB per-file limit. Each upload is treated as untrusted, checked
against its declared type before downstream processing, stored in an isolated
temporary run workspace, and assigned an exact-byte SHA-256 integrity hash.
Successful validation only confirms that a file is acceptable for future
processing; it does not mean that text has been extracted, indexed, or answered.

The current deterministic extraction layer converts accepted files into
citation-ready text sections without AI or network calls. PDF text is extracted
page by page when a text layer exists; V1 does not OCR scanned or image-only
pages. DOCX extraction captures paragraph text and heading styles only, ignoring
unsupported complex objects. TXT and Markdown use strict UTF-8, conservative
normalization, and source locators such as heading labels, line ranges, or PDF
page numbers.

Extracted sections are then chunked deterministically for future retrieval. V1
uses a configurable character-based policy (3,000 characters maximum and 300
characters of overlap by default), preserving section boundaries before splitting
at paragraphs, then simple sentence boundaries, and finally fixed character
positions. Chunks retain their parent locator and stable IDs; no AI changes their
source text.

The implemented retrieval layer is local and in-memory only. Its small
embedding-provider interface (`embed_documents` and `embed_query`) now has an
OpenAI implementation. When explicitly configured at runtime, exact document
chunk text is sent to the configured OpenAI embedding model and each question is
sent for one query embedding. The resulting vectors are used only by the local
NumPy cosine-similarity ranker; no persistent vector index, database, hosted
vector service, answer generation, chat, or completions behavior exists.

Set `OPENAI_API_KEY` server-side (for example, in a local ignored `.env` file or
a deployment secret manager). `OPENAI_EMBEDDING_MODEL` is optional and defaults
to `text-embedding-3-small`. Never expose either value to browser code or commit
credentials. This is an external-processing boundary: clients should understand
that chunk text and questions are sent to OpenAI when this provider is used.
Automated tests inject fake clients, make no OpenAI calls, and keep the suite-wide
network guard active. Indexes are never persisted. Each result retains document,
section, chunk, and path-free source locator IDs, so a future answer layer can
validate a citation against an actual retrieved chunk.

Answers are grounded in the uploaded materials only. Each material claim is
shown with visible source citations, so a user can inspect the document section
that supports it. When the materials do not support an answer, the assistant
will say so clearly instead of inventing policy or business information.

The included Harbor & Hearth Café documents are fictional demo data used to
demonstrate supported answers, multi-document answers, source traceability, and
unsupported-question handling.

## Grounded answers and citations

The grounded answer service accepts a question and already-retrieved candidate
sources; it never performs retrieval itself. Its injected provider receives only
a bounded, rank-ordered set of selected excerpts and must return structured JSON
with a status, answer, and chunk-ID citations. The application resolves each
citation locally against the current retrieved source set. Display document
names, locators, excerpts, and ranks come from those sources, never the model.

A supported answer requires at least one valid citation. Missing, malformed, or
unretrieved citations, malformed output, and empty retrieval all return the
stable no-support state rather than model text. Traceability verifies that a
citation names a retrieved excerpt; V1 does not claim perfect semantic
entailment. Source excerpts are treated as untrusted data, not instructions.

The optional `OpenAIAnswerProvider` uses the official Responses API with strict
JSON Schema, one request at most per supported query, no retries, and
`OPENAI_ANSWER_MODEL` (default `gpt-5.6-luna`). Retrieved excerpts and the
question are sent to OpenAI; no filesystem paths, vectors, unrelated documents,
or credentials are sent. There is no general-knowledge answering, chat memory,
or multi-turn state. Tests use fake providers and remain offline.

## Local demo interface

The optional Flask interface presents the same grounded workflow for a local
demo: add documents, prepare a temporary in-memory knowledge workspace, ask a
question, and inspect validated source cards. Start it after installing project
dependencies with:

```bash
.venv/bin/python -m ai_knowledge_assistant.web
```

It listens on `127.0.0.1:5000` by default. For an explicitly controlled local
Codespaces preview, use
`AI_KNOWLEDGE_ASSISTANT_HOST=0.0.0.0 .venv/bin/python -m ai_knowledge_assistant.web`;
do not expose it publicly without production security work.

Choose **Use Harbor & Hearth Demo** to process the six committed synthetic
documents through the same secure upload, extraction, chunking, and indexing
pipeline used for client uploads. Client uploads accept PDF, DOCX, TXT, and
Markdown files up to 10 MiB each. The browser cookie stores only an opaque run
identifier; documents, process-local indexes, and question/answer presentation
state remain in a temporary server-side workspace and **Reset workspace**
removes that run. When OpenAI providers are configured, document chunks and questions cross the existing OpenAI processing boundary;
templates and browser JavaScript never receive API keys, paths, vectors, or raw
provider failures.

### Screenshots

Screenshot placeholders for the landing screen, demo-ready workspace, supported
answer, and unsupported-answer state will be added after a local visual review.

## Deliberate V1 boundary

V1 is not:

- a general-purpose chatbot;
- an autonomous agent;
- a production multi-tenant SaaS product;
- a document-management platform; or
- an enterprise search deployment.

It also excludes enterprise integrations, SSO, complex roles and permissions,
voice or mobile experiences, huge document libraries, permanent client-document
storage, and advanced operational/compliance requirements. Those are
expanded-scope items and should be quoted separately.

## Repository safety

Runtime uploads, extracted text, indexes, local databases, generated artifacts,
caches, virtual environments, and environment files are ignored by Git. Use
`.env.example` as a keyless configuration template; keep real values only in a
local `.env` or an approved deployment secret manager.

Client documents are not permanent V1 storage. Accepted files live only in a
temporary, random run directory and should be explicitly cleaned up after use.

See [scope documentation](docs/scope.md), [privacy and data handling](docs/privacy-and-data-handling.md), [architecture](docs/architecture.md), and the [demo script](docs/demo-script.md).

## Current status

This foundation contains repository safety rules, deterministic document intake,
extraction, chunking, local retrieval, grounded answer generation, an optional
Flask local/demo interface, optional OpenAI providers, documentation, synthetic
demo documents, package metadata, and tests. It does not include persistent chat
or production SaaS behavior.
