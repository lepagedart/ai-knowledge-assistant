from __future__ import annotations

import hashlib
from pathlib import Path

from ai_knowledge_assistant.chunking import (
    CHUNKING_VERSION,
    ChunkingConfig,
    chunk_document,
)
from ai_knowledge_assistant.models import (
    DocumentType,
    ExtractedDocument,
    ExtractedSection,
    SourceLocator,
    SourceLocatorKind,
)


def _document(*texts: str) -> ExtractedDocument:
    sections = tuple(
        ExtractedSection(
            section_id=f"section-{index}",
            document_id="document-1",
            document_display_name="fictional-policy.md",
            document_type=DocumentType.MARKDOWN,
            source_locator=SourceLocator(
                kind=SourceLocatorKind.DOCUMENT_SECTION,
                section_label=f"Section {index}",
                heading_level=1,
            ),
            text=text,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        for index, text in enumerate(texts)
    )
    return ExtractedDocument(
        extraction_version="v1",
        document_id="document-1",
        document_display_name="fictional-policy.md",
        document_type=DocumentType.MARKDOWN,
        source_content_hash="source-hash",
        sections=sections,
    )


def test_short_section_produces_one_citation_preserving_chunk() -> None:
    document = _document("Keep this exact source text.")

    chunks = chunk_document(
        document, ChunkingConfig(max_characters=100, overlap_characters=10)
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Keep this exact source text."
    assert chunks[0].section_id == "section-0"
    assert chunks[0].source_locator == document.sections[0].source_locator
    assert chunks[0].source_char_start == 0
    assert chunks[0].source_char_end == len(document.sections[0].text)
    assert chunks[0].chunking_version == CHUNKING_VERSION


def test_long_sections_prefer_paragraph_boundaries_and_limit_chunk_size() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    document = _document(text)

    chunks = chunk_document(
        document, ChunkingConfig(max_characters=28, overlap_characters=4)
    )

    assert chunks[0].text == "First paragraph.\n\n"
    assert all(len(chunk.text) <= 28 for chunk in chunks)
    assert _reconstruct_primary_text(chunks) == text


def test_long_single_paragraph_uses_character_fallback_and_overlap() -> None:
    text = "x" * 23
    document = _document(text)

    chunks = chunk_document(
        document, ChunkingConfig(max_characters=10, overlap_characters=3)
    )

    assert [chunk.text for chunk in chunks] == [
        "x" * 10,
        "x" * 10,
        "x" * 9,
    ]
    assert [chunk.source_char_start for chunk in chunks] == [0, 7, 14]
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert _reconstruct_primary_text(chunks) == text


def test_sections_at_the_size_limit_are_not_split_or_overlapped() -> None:
    document = _document("a" * 12)

    chunks = chunk_document(
        document, ChunkingConfig(max_characters=12, overlap_characters=3)
    )

    assert len(chunks) == 1
    assert chunks[0].text == "a" * 12


def test_empty_sections_are_skipped_without_affecting_order() -> None:
    document = _document("", "   ", "Useful section.")

    chunks = chunk_document(document)

    assert [chunk.section_id for chunk in chunks] == ["section-2"]


def test_unicode_and_heading_only_sections_are_preserved_exactly() -> None:
    document = _document("Résumé 🌱", "Safety")

    chunks = chunk_document(document)

    assert [chunk.text for chunk in chunks] == ["Résumé 🌱", "Safety"]


def test_chunk_ids_are_stable_and_change_with_meaningful_config_change() -> None:
    document = _document("word " * 20)
    first_config = ChunkingConfig(max_characters=20, overlap_characters=2)
    second_config = ChunkingConfig(max_characters=16, overlap_characters=2)

    first = chunk_document(document, first_config)
    repeated = chunk_document(document, first_config)
    changed = chunk_document(document, second_config)

    assert first == repeated
    assert first[0].chunk_id != changed[0].chunk_id


def test_content_hashes_and_section_lineage_match_exact_chunk_text() -> None:
    document = _document("Hash this exact text.")

    chunk = chunk_document(document)[0]

    assert chunk.document_id == document.document_id
    assert chunk.document_name == document.document_display_name
    assert chunk.source_section_content_hash == document.sections[0].content_hash
    assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def test_chunking_does_not_expose_absolute_paths_or_read_files(tmp_path: Path) -> None:
    document = _document("No filesystem read is needed.")

    chunks = chunk_document(document)

    assert str(tmp_path) not in repr(chunks)
    assert chunks[0].text == "No filesystem read is needed."


def _reconstruct_primary_text(chunks: tuple) -> str:
    return "".join(
        chunk.text[
            chunk.primary_char_start - chunk.source_char_start : chunk.primary_char_end
            - chunk.source_char_start
        ]
        for chunk in chunks
    )
