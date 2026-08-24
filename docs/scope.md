# V1 Scope

## Structured records V1

CSV/XLSX deterministic ingestion supports invoice, purchase order, vendor,
product catalog, and generic rows with retrieval and grounded citations.
Financial calculation, reconciliation, accounting, recommendations, analytics,
OCR, integrations, and autonomous actions remain deferred.

## Standard implementation — starting at $550

The standard implementation is a focused reference build for one business and
one business knowledge assistant. It supports a reasonably sized document set,
the standard V1 formats (PDF, DOCX, TXT, and Markdown), grounded questions and
answers with source citations, and basic local setup and testing.

The intended outcome is a polished, inspectable demonstration of how internal
SOPs, policies, training material, handbooks, and reference guides can answer
staff questions without presenting unsupported information as fact.

## Included boundary

- One business knowledge base and one assistant experience.
- Deterministic document ingestion, text extraction, chunking, and local
  in-memory vector retrieval through an injectable embedding-provider contract;
  optional OpenAI embeddings send chunk text and questions to OpenAI while
  similarity ranking remains local.
- Citation-ready source locations for extracted sections, such as PDF pages and
  document headings.
- Answers grounded only in supplied materials, with visible source references.
- Clear unsupported-answer handling when the materials do not establish an answer.
- A small synthetic demo corpus, project documentation, and baseline tests.

Retrieval returns ranked citation-ready candidate evidence only. Its cosine
similarity scores are not factual confidence and do not generate answers; a
future grounded answer layer must validate support against retrieved chunk IDs.
V1 has no answer-generation AI, hosted vector service, database, or persistent
client index. OpenAI embedding requests occur only when the optional provider is
explicitly configured and called; automated tests remain fully offline.

## Grounded answer behavior

The answer layer receives only selected retrieved evidence through an injected
provider, then validates structured chunk-ID citations locally. Missing evidence
or invalid output produces an explicit no-support state. It does not use general
knowledge, and citation traceability does not guarantee perfect semantic
entailment. V1 has no chat memory or multi-turn state; tests remain offline.

## Local demo interface

The included Flask interface is a single-process local/demo presentation layer,
not a production multi-user application. It uses one temporary workspace per
browser session, a process-local index, server-side question/answer presentation
state, explicit reset cleanup, and the existing secure ingestion/retrieval/answer
boundaries. Its browser cookie holds only the opaque workspace identifier, apart
from normal transient Flask framework state. It has no authentication, database,
persistent files, persistent conversations, or hosted UI deployment.

Its presentation uses a local-first typography system: a restrained classic
display serif (Iowan Old Style/Palatino fallbacks) gives business headings a
measured hospitality character, while the operating-system UI sans stack keeps
controls, documents, and answers highly readable without loading font assets.
The sidebar uses an original inline SVG martini/chart mark and a unified gold
Raise the Bar Consulting wordmark; it has no external image or font dependency.

## Expanded scope, quoted separately

Examples include large archives or bulk migration, scanned-document OCR,
additional file types, Google Drive/Notion/SharePoint/Slack integrations,
accounts, SSO, multi-tenancy, fine-grained permissions, permanent cloud storage,
production deployment and observability, compliance/security review, custom
design systems, analytics, voice, mobile applications, multilingual behavior,
and ongoing maintenance.

### Potential future capability: structured business records

Invoices, purchase orders, vendor records, and product/catalog records are not
V1 evidence. Supporting them would require format-specific ingestion,
deterministic structured-data validation, and appropriate business logic before
they could become assistant evidence.
