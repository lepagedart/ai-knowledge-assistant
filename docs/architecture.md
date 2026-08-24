# Planned V1 Architecture

```text
Ingestion → extraction → chunking → embeddings → retrieval → grounded answer
```

The future implementation separates each stage behind small provider interfaces.
Ingestion validates and isolates untrusted uploads in a UUID-named temporary
workspace. It records exact-byte SHA-256 hashes and generated document IDs;
client filenames are display metadata only and never select a filesystem path.
The implemented extraction layer then deterministically produces text and source
location metadata from the accepted bytes only. PDF sections retain 1-based page
numbers, DOCX and Markdown retain heading labels where present, and TXT retains
line ranges. The implemented chunking layer preserves each section as a primary
semantic boundary and uses configurable character limits (3,000 maximum and 300
overlap by default). It prefers paragraph boundaries, then simple sentence
boundaries, and uses fixed character positions only as a fallback. Chunks retain
parent locators, exact source character ranges, stable IDs, and content hashes.
An embeddings provider will later consume those chunks without altering them, and
a retrieval provider will rank local chunks with NumPy cosine similarity.

Hash roles are intentionally distinct: the accepted document SHA-256 identifies
the original file bytes; an extracted section hash identifies its normalized
extracted text; a chunk content hash identifies that chunk’s exact text; and a
chunk ID additionally incorporates its source section, character ranges,
chunking version, and configuration.

Extraction makes no network calls and no AI calls. It does not OCR PDFs, render
Markdown or HTML, execute DOCX macros, load embedded/external resources, or
invent missing document content. Empty or image-only documents produce a stable
no-extractable-text state rather than fabricated text.

Chunking performs no filesystem reads, network calls, or AI calls. It skips empty
sections and never merges unrelated source sections merely to meet a target size.

V1 does not need a standalone vector database: a reasonably sized, single-
business document set can use local vectors and explicit chunk metadata. This
keeps retrieval inspectable and deterministic in tests by substituting a fixed
embedding provider.

The answer provider receives only the retrieved excerpts and must return a
structured answer with cited chunk identifiers. Citation validation resolves each
identifier back to an actual retrieved chunk before the UI presents it. If there
is no sufficiently relevant material, or citations cannot be validated, the
application returns an unsupported-answer state rather than an invented answer.
