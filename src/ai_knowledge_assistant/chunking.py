"""Deterministic, citation-preserving segmentation of extracted documents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .models import DocumentChunk, ExtractedDocument, ExtractedSection

CHUNKING_VERSION = "v1"
DEFAULT_MAX_CHARACTERS = 3_000
DEFAULT_OVERLAP_CHARACTERS = 300
_SENTENCE_BOUNDARY = re.compile(r"[.!?][\]\"')]*\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based V1 chunk sizing configuration."""

    max_characters: int = DEFAULT_MAX_CHARACTERS
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters must be positive.")
        if not 0 <= self.overlap_characters < self.max_characters:
            raise ValueError(
                "overlap_characters must be nonnegative and less than "
                "max_characters."
            )


def chunk_document(
    extracted_document: ExtractedDocument,
    config: ChunkingConfig | None = None,
) -> tuple[DocumentChunk, ...]:
    """Create ordered chunks without reading files or changing source text.

    ``source_char_*`` marks the complete text range in the parent section,
    including configured overlap. ``primary_char_*`` marks the non-overlapping
    source range, which can reconstruct the original section when concatenated
    in chunk order.
    """
    active_config = config or ChunkingConfig()
    chunks: list[DocumentChunk] = []
    for section in extracted_document.sections:
        if not section.text.strip():
            continue
        for chunk_index, (start, end, primary_start, primary_end) in enumerate(
            _chunk_ranges(section.text, active_config)
        ):
            text = section.text[start:end]
            chunks.append(
                _chunk(
                    extracted_document,
                    section,
                    chunk_index,
                    text,
                    start,
                    end,
                    primary_start,
                    primary_end,
                    active_config,
                )
            )
    return tuple(chunks)


def _chunk_ranges(
    text: str, config: ChunkingConfig
) -> tuple[tuple[int, int, int, int], ...]:
    primary_ranges = _primary_ranges(
        text, config.max_characters, config.overlap_characters
    )
    ranges: list[tuple[int, int, int, int]] = []
    for index, (primary_start, primary_end) in enumerate(primary_ranges):
        start = primary_start
        if index:
            start = max(0, primary_start - config.overlap_characters)
        ranges.append((start, primary_end, primary_start, primary_end))
    return tuple(ranges)


def _primary_ranges(
    text: str, maximum: int, overlap: int
) -> tuple[tuple[int, int], ...]:
    """Split source text by paragraphs, then sentences, then character limit."""
    if len(text) <= maximum:
        return ((0, len(text)),)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        primary_limit = maximum if not ranges else maximum - overlap
        maximum_end = min(start + primary_limit, len(text))
        if maximum_end == len(text):
            end = len(text)
        else:
            end = _preferred_split_end(text, start, maximum_end)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _preferred_split_end(text: str, start: int, maximum_end: int) -> int:
    paragraph_end = text.rfind("\n\n", start + 1, maximum_end + 1)
    if paragraph_end != -1:
        return paragraph_end + 2

    sentence_end = _last_sentence_boundary(text, start, maximum_end)
    if sentence_end is not None:
        return sentence_end
    return maximum_end


def _last_sentence_boundary(text: str, start: int, maximum_end: int) -> int | None:
    result: int | None = None
    for match in _SENTENCE_BOUNDARY.finditer(text, start, maximum_end):
        result = match.end()
    return result


def _chunk(
    document: ExtractedDocument,
    section: ExtractedSection,
    chunk_index: int,
    text: str,
    start: int,
    end: int,
    primary_start: int,
    primary_end: int,
    config: ChunkingConfig,
) -> DocumentChunk:
    config_key = json.dumps(
        {
            "max_characters": config.max_characters,
            "overlap_characters": config.overlap_characters,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    chunk_key = "\x1f".join(
        (
            CHUNKING_VERSION,
            document.document_id,
            section.section_id,
            config_key,
            str(chunk_index),
            str(start),
            str(end),
            str(primary_start),
            str(primary_end),
            text,
        )
    )
    return DocumentChunk(
        chunk_id=hashlib.sha256(chunk_key.encode("utf-8")).hexdigest(),
        document_id=document.document_id,
        document_name=document.document_display_name,
        document_type=document.document_type,
        section_id=section.section_id,
        chunk_index=chunk_index,
        text=text,
        source_locator=section.source_locator,
        source_char_start=start,
        source_char_end=end,
        primary_char_start=primary_start,
        primary_char_end=primary_end,
        source_section_content_hash=section.content_hash,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chunking_version=CHUNKING_VERSION,
    )
