# Privacy and Data Handling

## V1 handling model

V1 is designed to use a temporary, isolated workspace for each upload or demo
run. Accepted files, extracted text, chunk metadata, and local search/index
artifacts belong inside that workspace rather than a shared, permanent client
library.

The V1 upload boundary accepts PDF, DOCX, TXT, and Markdown only, with a
10 MiB per-file limit. It validates a PDF signature, expected DOCX ZIP members,
or strict UTF-8 text as appropriate, and records a SHA-256 digest of the exact
accepted bytes. This is an input-safety and provenance check, not text
extraction, malware detection, OCR, semantic review, or indexing.

Client files must never be committed to this public repository. The repository
contains only fictional Harbor & Hearth Café material for demonstration and
testing. Runtime upload folders, indexes, local databases, caches, and generated
artifacts are ignored by Git.

## Credentials and providers

API keys are supplied through environment variables or a deployment secret
manager, never browser code, source files, fixtures, logs, or commits.
`.env.example` is deliberately keyless; local `.env` files are ignored.

When the OpenAI-backed answer and embedding layers are implemented, relevant
uploaded text and user questions will be sent to OpenAI for that processing.
Clients should review the applicable OpenAI service and data-handling terms
before using the implementation with non-demo information.

## Retention and deletion expectations

The implementation should provide a defined temporary retention period and a
clear deletion path for each workspace. Local/demo V1 is not a document archive,
records-management system, or compliance storage solution. It does not promise
enterprise retention controls, legal holds, backups, audit trails, residency
guarantees, or production security certification.
