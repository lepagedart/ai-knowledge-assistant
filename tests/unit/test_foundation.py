"""Foundation-level repository tests."""

from pathlib import Path

import ai_knowledge_assistant
from ai_knowledge_assistant.web import DEMO_DIRECTORY

ROOT = Path(__file__).resolve().parents[2]
DEMO_DOCUMENTS = DEMO_DIRECTORY
REQUIRED_DOCUMENTS = {
    "employee_handbook.md",
    "opening_closing_sop.md",
    "callout_attendance_policy.md",
    "refund_service_recovery_policy.md",
    "menu_product_reference.md",
    "new_team_member_training_guide.md",
}
DEMO_FOOTER = "Demo content only — no real business information."


def test_package_imports() -> None:
    assert ai_knowledge_assistant.__version__ == "0.1.0"


def test_required_documentation_exists() -> None:
    for relative_path in (
        "README.md",
        "docs/scope.md",
        "docs/privacy-and-data-handling.md",
        "docs/demo-script.md",
        "docs/architecture.md",
    ):
        assert (ROOT / relative_path).is_file()


def test_demo_documents_are_complete_and_labeled() -> None:
    assert {path.name for path in DEMO_DOCUMENTS.glob("*.md")} == REQUIRED_DOCUMENTS
    for name in REQUIRED_DOCUMENTS:
        assert DEMO_FOOTER in (DEMO_DOCUMENTS / name).read_text(encoding="utf-8")


def test_demo_documents_do_not_contain_ceo_home_address_information() -> None:
    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in DEMO_DOCUMENTS.glob("*.md")
    ).lower()
    assert "ceo" not in corpus
    assert "home address" not in corpus
