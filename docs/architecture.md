# Planned V1 Architecture

```text
Ingestion → extraction → chunking → embeddings → retrieval → grounded answer
```

The future implementation separates each stage behind small provider interfaces.
Ingestion validates and isolates untrusted uploads in a UUID-named temporary
workspace. It records exact-byte SHA-256 hashes and generated document IDs;
client filenames are display metadata only and never select a filesystem path.
Extraction then produces text with source location metadata. Chunking creates
stable, inspectable chunks. An embeddings provider creates vectors, and a
retrieval provider ranks local chunks with NumPy cosine similarity.

V1 does not need a standalone vector database: a reasonably sized, single-
business document set can use local vectors and explicit chunk metadata. This
keeps retrieval inspectable and deterministic in tests by substituting a fixed
embedding provider.

The answer provider receives only the retrieved excerpts and must return a
structured answer with cited chunk identifiers. Citation validation resolves each
identifier back to an actual retrieved chunk before the UI presents it. If there
is no sufficiently relevant material, or citations cannot be validated, the
application returns an unsupported-answer state rather than an invented answer.
