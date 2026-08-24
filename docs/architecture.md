# Planned V1 Architecture

```text
Ingestion → extraction → chunking → embeddings → retrieval → grounded answer
```

The implementation separates each stage behind small provider interfaces.
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
The retrieval boundary has an `EmbeddingProvider` protocol with
`embed_documents(texts)` and `embed_query(text)` methods; no OpenAI provider is
implemented yet. `build_index` passes exact immutable chunk text to that provider
and constructs a process-local `LocalVectorIndex`. `retrieve` trims and validates
the question, embeds it through the injected provider, then ranks records by
cosine similarity: dot(query, document) / (norm(query) * norm(document)). Empty,
zero, non-finite, and dimension-mismatched vectors are rejected. Equal scores
are ordered by original chunk build order, making tie handling deterministic.

The default minimum score of 0.2 is a conservative V1 candidate filter and is
applied after scoring. A score is a ranking signal, never a factual-confidence
percentage or proof that a source supports an answer. No qualifying entries
returns an explicit empty result; a later answer layer remains responsible for
grounded support or refusal.

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
business document set uses NumPy-backed in-memory vectors and explicit chunk
metadata. No vector index is persisted. `EmbeddedChunk` and `RetrievedSource`
retain document → section → chunk → source-locator lineage and exact text/content
hashes without filesystem paths. A future answer service can validate a proposed
citation using retrieved chunk IDs alone. Tests substitute a deterministic,
transparent lexical embedding provider; production code makes no network calls.

The answer provider receives only the retrieved excerpts and must return a
structured answer with cited chunk identifiers. Citation validation resolves each
identifier back to an actual retrieved chunk before the UI presents it. If there
is no sufficiently relevant material, or citations cannot be validated, the
application returns an unsupported-answer state rather than an invented answer.
