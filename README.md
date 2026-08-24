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

Answers are grounded in the uploaded materials only. Each material claim is
shown with visible source citations, so a user can inspect the document section
that supports it. When the materials do not support an answer, the assistant
will say so clearly instead of inventing policy or business information.

The included Harbor & Hearth Café documents are fictional demo data used to
demonstrate supported answers, multi-document answers, source traceability, and
unsupported-question handling.

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

This foundation contains repository safety rules, documentation, synthetic demo
documents, package metadata, and tests. It intentionally does not yet include
web routes, OpenAI calls, embeddings, retrieval, or chat behavior.
