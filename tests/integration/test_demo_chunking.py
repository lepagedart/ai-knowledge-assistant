from pathlib import Path

from ai_knowledge_assistant.chunking import chunk_document
from ai_knowledge_assistant.extraction import extract_document
from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.web import DEMO_DIRECTORY
from ai_knowledge_assistant.workspace import UploadWorkspace


def test_harbor_and_hearth_documents_flow_through_ingestion_extraction_and_chunking(
    tmp_path: Path,
) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    chunks_by_name = {}
    for source_path in sorted(DEMO_DIRECTORY.glob("*.md")):
        accepted = accept_upload(
            workspace, source_path.name, source_path.read_bytes()
        )
        extracted = extract_document(workspace, accepted)
        chunks_by_name[source_path.name] = chunk_document(extracted)

    all_text = "\n".join(
        chunk.text for chunks in chunks_by_name.values() for chunk in chunks
    ).lower()

    assert "at least two hours before the shift starts" in _text(
        chunks_by_name["callout_attendance_policy.md"]
    )
    assert "opening float must equal $200.00" in _text(
        chunks_by_name["opening_closing_sop.md"]
    )
    assert "allergen-related concern require manager review" in _text(
        chunks_by_name["refund_service_recovery_policy.md"]
    )
    assert "new team members complete an orientation" in _text(
        chunks_by_name["new_team_member_training_guide.md"]
    ).lower()
    assert "ceo" not in all_text
    assert "home address" not in all_text


def _text(chunks: tuple) -> str:
    return "\n".join(chunk.text for chunk in chunks)
