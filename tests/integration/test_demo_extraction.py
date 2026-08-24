from pathlib import Path

from ai_knowledge_assistant.extraction import extract_document
from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.workspace import UploadWorkspace

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIRECTORY = ROOT / "demo_documents" / "harbor_and_hearth"


def test_harbor_and_hearth_markdown_documents_extract_with_source_headings(
    tmp_path: Path,
) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    extracted_by_name = {}
    for source_path in sorted(DEMO_DIRECTORY.glob("*.md")):
        accepted = accept_upload(
            workspace, source_path.name, source_path.read_bytes()
        )
        extracted_by_name[source_path.name] = extract_document(workspace, accepted)

    callout = extracted_by_name["callout_attendance_policy.md"]
    opening = extracted_by_name["opening_closing_sop.md"]
    refund = extracted_by_name["refund_service_recovery_policy.md"]
    menu = extracted_by_name["menu_product_reference.md"]
    all_text = "\n".join(
        section.text
        for document in extracted_by_name.values()
        for section in document.sections
    ).lower()

    assert "Calling out" in [
        section.source_locator.section_label for section in callout.sections
    ]
    assert "at least two hours before the shift starts" in "\n".join(
        section.text for section in callout.sections
    )
    assert "Opening the cash drawer" in [
        section.source_locator.section_label for section in opening.sections
    ]
    assert "opening float must equal $200.00" in "\n".join(
        section.text for section in opening.sections
    )
    assert "allergen-related concern require manager review" in "\n".join(
        section.text for section in refund.sections
    )
    assert "Allergen concern escalation" in [
        section.source_locator.section_label for section in menu.sections
    ]
    assert "ceo" not in all_text
    assert "home address" not in all_text
