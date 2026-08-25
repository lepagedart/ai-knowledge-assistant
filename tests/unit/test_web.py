"""Offline Flask test-client coverage for the local knowledge-assistant UI."""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Sequence

from ai_knowledge_assistant.models import (
    GroundedAnswer,
    GroundedAnswerStatus,
    GroundedCitation,
    ProviderAnswer,
    ProviderCitation,
    SourceLocator,
    SourceLocatorKind,
)
from ai_knowledge_assistant.reconciliation import reconcile
from ai_knowledge_assistant.structured_records import parse_structured_document
from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.web import (
    _answer_view,
    _reconciliation_view,
    create_app,
    document_display_title,
)
from ai_knowledge_assistant.workspace import UploadWorkspace


class FakeEmbeddingProvider:
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple((1.0, float(index + 1)) for index, _ in enumerate(texts))

    def embed_query(self, text: str) -> Sequence[float]:
        if "CEO" in text:
            return (-1.0, 0.0)
        return (1.0, 1.0)


class FakeAnswerProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def generate_answer(
        self, question: str, sources: tuple[object, ...]
    ) -> ProviderAnswer:
        del question
        self.calls += 1
        if self.fail:
            raise RuntimeError("private provider output")
        return ProviderAnswer(
            "supported",
            "Grounded operating guidance.",
            (ProviderCitation(sources[0].chunk_id, "Grounded guidance."),),  # type: ignore[attr-defined]
        )


def _app(tmp_path: Path, answer_provider: FakeAnswerProvider | None = None):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "WORKSPACE_ROOT": tmp_path,
            "EMBEDDING_PROVIDER": FakeEmbeddingProvider(),
            "ANSWER_PROVIDER": answer_provider or FakeAnswerProvider(),
        }
    )


def _upload(
    client: object,
    name: str = "guide.md",
    content: bytes = b"# Guide\nPolicy text.",
):
    return client.post(  # type: ignore[attr-defined]
        "/upload",
        data={"documents": [(BytesIO(content), name)]},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


INVOICE_HEADER = (
    "Invoice Number,PO Number,Vendor,SKU,Item,Quantity,Unit Price,Line Total,Unit\n"
)
PO_HEADER = "PO Number,Vendor,SKU,Item,Quantity,Unit Price,Unit\n"


def _reconcile_csvs(tmp_path: Path, invoices: str, purchase_orders: str):
    workspace = UploadWorkspace.create(tmp_path)
    invoice = accept_upload(workspace, "invoices.csv", invoices.encode())
    purchase_order = accept_upload(
        workspace, "purchase_orders.csv", purchase_orders.encode()
    )
    records = tuple(
        record
        for document, content in (
            (invoice, invoices),
            (purchase_order, purchase_orders),
        )
        for sheet in parse_structured_document(document, content.encode()).sheets
        for record in sheet.records
    )
    return reconcile(records)


def test_landing_upload_and_demo_workflows_are_safe_and_ready(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()

    landing = client.get("/")
    uploaded = _upload(client)
    demo = client.post("/demo", follow_redirects=True)

    assert b"Turn your business documents" in landing.data
    assert b"guide.md" in uploaded.data
    assert b"Indexed / ready" in uploaded.data
    assert b"11 documents" in demo.data
    assert b"Synthetic Harbor &amp; Hearth demo" in demo.data
    assert str(tmp_path).encode() not in demo.data
    assert b"test-secret" not in demo.data


def test_structured_demo_files_and_citation_card_render_without_internal_ids(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    response = client.post("/demo", follow_redirects=True)
    run = next(iter(app.extensions["knowledge_runs"].values()))
    run.answer_view = _answer_view(
        GroundedAnswer(
            GroundedAnswerStatus.SUPPORTED,
            "INV-1048 records $147.00 for London Dry Gin.",
            (
                GroundedCitation(
                    "internal-structured-id",
                    "Invoice evidence.",
                    "harbor_hearth_invoices.csv",
                    SourceLocator(
                        SourceLocatorKind.STRUCTURED_ROW,
                        row_number=2,
                        record_label="INV-1048",
                    ),
                    "Record type: Invoice\nInvoice Number: INV-1048",
                    1,
                ),
            ),
            ("internal-structured-id",),
        )
    )
    rendered = client.get("/")

    with client.session_transaction() as browser_session:
        session_data = dict(browser_session)

    assert b"Harbor &amp; Hearth Invoices (fictional)" in response.data
    assert b"Harbor &amp; Hearth Products (fictional)" in response.data
    assert b"Invoice INV-1048" not in rendered.data  # label is intentionally compact
    assert b"INV-1048" in rendered.data
    assert b"Row 2" in rendered.data
    assert b"internal-structured-id" not in rendered.data
    assert b"chunk_id" not in rendered.data
    assert session_data.keys() == {"run_id"}
    assert "INV-1048" not in str(session_data)


def test_landing_renders_the_inline_gold_brand_mark_without_external_images(
    tmp_path: Path,
) -> None:
    response = _app(tmp_path).test_client().get("/")

    assert b'<svg class="brand-mark"' in response.data
    assert b'aria-hidden="true"' in response.data
    assert b"RAISE THE BAR" in response.data
    assert b"CONSULTING" in response.data
    assert b"<img" not in response.data
    assert b"http://" not in response.data
    assert b"https://" not in response.data


def test_landing_includes_an_accessible_mobile_workspace_menu(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    response = client.post("/demo", follow_redirects=True)

    assert b'class="menu-toggle"' in response.data
    assert b'aria-expanded="false"' in response.data
    assert b'aria-controls="workspace-navigation"' in response.data
    assert b"Workspace menu" in response.data
    assert b'id="workspace-navigation"' in response.data
    assert b">Documents<" in response.data
    assert b">Ask<" in response.data
    assert b">Sources<" in response.data
    assert b"Reset workspace" in response.data


def test_demo_reconciliation_uses_only_intended_sources_and_counts(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    response = app.test_client().post("/demo", follow_redirects=True)
    run = next(iter(app.extensions["knowledge_runs"].values()))
    summary = run.reconciliation.summary

    assert summary.matched_line_count == 1
    assert summary.variance_line_count == 2
    assert summary.missing_on_po_count == 1
    assert summary.missing_on_invoice_count == 3
    assert summary.total_monetary_variance == Decimal("21.00")
    assert summary.matched_line_count == sum(
        line.status.value == "MATCHED" for line in run.reconciliation.lines
    )
    assert summary.variance_line_count == sum(
        line.status.value == "VARIANCE" for line in run.reconciliation.lines
    )
    assert summary.missing_on_po_count == sum(
        line.status.value == "MISSING_ON_PO" for line in run.reconciliation.lines
    )
    assert summary.missing_on_invoice_count == sum(
        line.status.value == "MISSING_ON_INVOICE" for line in run.reconciliation.lines
    )
    assert all(
        not line.purchase_order
        or line.purchase_order.document_name
        != "harbor_hearth_purchase_orders.xlsx"
        for line in run.reconciliation.lines
    )
    assert b"Needs attention" in response.data
    assert b"$21.00" in response.data


def test_reconciliation_groups_ambiguity_candidates_and_keeps_full_audit_disclosable(
    tmp_path: Path,
) -> None:
    response = _app(tmp_path).test_client().post("/demo", follow_redirects=True)
    default_cards, audit_cards = response.data.split(b'id="all-reconciliation-lines"')

    assert b"Price variance" in default_cards
    assert b"Quantity variance" in default_cards
    assert b"Not found on purchase order" in default_cards
    assert b"Ordered but not invoiced" in default_cards
    assert b"Unit mismatch" in default_cards
    assert b"Ambiguous match" in default_cards
    assert b"Tonic Water" not in default_cards
    assert default_cards.count(b"House Bitters") == 1
    assert b"2 purchase-order lines could match this invoice line" in default_cards
    assert b"Ice Cubes" in default_cards
    assert b"Needs attention</span><strong>6" in default_cards
    assert b"Show 9 deterministic audit lines" in response.data
    assert b'aria-expanded="false"' in response.data
    assert b'aria-controls="all-reconciliation-lines"' in response.data
    assert (
        b'<div id="all-reconciliation-lines" class="citation-grid '
        b'reconciliation-grid matched-lines" hidden>'
    ) in response.data
    assert b"Tonic Water" in audit_cards
    assert b"Matched</p><h4>Tonic Water</h4>" in audit_cards
    assert b"Needs review</p><h4>Tonic Water</h4>" not in audit_cards
    assert audit_cards.count(b"House Bitters") == 3
    assert b"Ambiguity candidate \xe2\x80\x94 ordered but not invoiced" in audit_cards


def test_reconciliation_disclosure_honors_hidden_contract() -> None:
    css = Path("src/ai_knowledge_assistant/static/app.css").read_text()
    javascript = Path("src/ai_knowledge_assistant/static/app.js").read_text()

    assert "[hidden]{display:none!important}" in css
    assert 'getAttribute("aria-controls")' in javascript
    assert "document.getElementById(reconciliationLinesId)" in javascript
    assert "matchedReconciliationLines.hidden = !expanded" in javascript
    assert 'setAttribute("aria-expanded", String(expanded))' in javascript


def test_all_exception_reconciliation_has_no_matched_engine_lines(
    tmp_path: Path,
) -> None:
    result = _reconcile_csvs(
        tmp_path,
        INVOICE_HEADER + "INV-1,PO-1,V,A,Extra,1,1,1,each\n",
        PO_HEADER + "PO-1,V,B,Absent,1,1,each\n",
    )
    view = _reconciliation_view(result)

    assert all(line["status"] != "Matched" for line in view["audit_lines"])
    assert view["exception_count"] == 2


def test_clean_reconciliation_has_a_dynamic_positive_state(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/upload",
        data={
            "documents": [
                (
                    BytesIO(
                        (INVOICE_HEADER + "INV-1,PO-1,V,A,Item,1,10,10,each\n").encode()
                    ),
                    "invoices.csv",
                ),
                (
                    BytesIO((PO_HEADER + "PO-1,V,A,Item,1,10,each\n").encode()),
                    "purchase_orders.csv",
                ),
            ]
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert b"All 1 lines reconciled" in response.data
    assert b"Show 1 deterministic audit line" in response.data
    assert b"Needs attention</span><strong>0" in response.data


def test_plain_english_exception_labels_are_derived_from_engine_outcomes(
    tmp_path: Path,
) -> None:
    price = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "price",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,1,12,12,each\n",
            PO_HEADER + "PO-1,V,A,Item,1,10,each\n",
        )
    )
    quantity = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "quantity",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,2,10,20,each\n",
            PO_HEADER + "PO-1,V,A,Item,1,10,each\n",
        )
    )
    unit = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "unit",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,1,10,10,case\n",
            PO_HEADER + "PO-1,V,A,Item,1,10,bottle\n",
        )
    )
    ambiguity = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "ambiguity",
            INVOICE_HEADER + "INV-1,PO-1,V,,Item,1,10,10,each\n",
            PO_HEADER + "PO-1,V,A,Item,1,10,each\nPO-1,V,B,Item,1,10,each\n",
        )
    )

    assert price["exception_lines"][0]["exception_label"] == "Price variance"
    assert quantity["exception_lines"][0]["exception_label"] == "Quantity variance"
    assert unit["exception_lines"][0]["exception_label"] == "Unit mismatch"
    assert ambiguity["exception_lines"][0]["exception_label"] == "Ambiguous match"


def test_reconciliation_projection_exposes_deterministic_comparison_details(
    tmp_path: Path,
) -> None:
    quantity = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "quantity-details",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,5,18,90,bag\n",
            PO_HEADER + "PO-1,V,A,Item,4,18,bag\n",
        )
    )["exception_lines"][0]
    price = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "price-details",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,6,24.50,147,each\n",
            PO_HEADER + "PO-1,V,A,Item,6,24,each\n",
        )
    )["exception_lines"][0]
    multiple = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "multiple-details",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,2,12,24,each\n",
            PO_HEADER + "PO-1,V,A,Item,1,10,each\n",
        )
    )["exception_lines"][0]
    unit = _reconciliation_view(
        _reconcile_csvs(
            tmp_path / "unit-details",
            INVOICE_HEADER + "INV-1,PO-1,V,A,Item,1,10,10,case\n",
            PO_HEADER + "PO-1,V,A,Item,1,10,bottle\n",
        )
    )["exception_lines"][0]

    assert quantity["quantity_ordered"] == "4"
    assert quantity["quantity_invoiced"] == "5"
    assert quantity["quantity_difference"] == "+1"
    assert quantity["extended_variance"] == "+$18.00"
    assert price["po_unit_price"] == "$24.00"
    assert price["invoice_unit_price"] == "$24.50"
    assert price["unit_price_difference"] == "+$0.50"
    assert price["extended_variance"] == "+$3.00"
    assert multiple["exception_label"] == "Quantity & price variance"
    assert multiple["quantity_difference"] == "+1"
    assert multiple["unit_price_difference"] == "+$2.00"
    assert unit["po_unit"] == "bottle"
    assert unit["invoice_unit"] == "case"
    assert unit["extended_variance"] is None


def test_reconciliation_cards_render_comparisons_without_internal_identifiers(
    tmp_path: Path,
) -> None:
    response = _app(tmp_path).test_client().post("/demo", follow_redirects=True)
    default_cards = response.data.split(b'id="all-reconciliation-lines"')[0]

    assert b"Ordered quantity: 4 bag" in default_cards
    assert b"Invoiced quantity: 5 bag" in default_cards
    assert b"Difference: +1 bag" in default_cards
    assert b"PO unit price: $24.00" in default_cards
    assert b"Invoice unit price: $24.50" in default_cards
    assert b"Difference: +$0.50 per unit" in default_cards
    assert b"PO unit: bottle" in default_cards
    assert b"Invoice unit: case" in default_cards
    assert b"No financial comparison performed" in default_cards
    assert b"reconciliation_id" not in response.data
    assert b"record_id" not in response.data
    assert b"ambiguity_candidate_po_record_ids" not in response.data


def test_reconciliation_groups_only_exact_ambiguity_candidates(tmp_path: Path) -> None:
    result = _reconcile_csvs(
        tmp_path,
        INVOICE_HEADER + "INV-1,PO-1,Vendor A,SKU-1,Item,1,10,10,each\n",
        PO_HEADER
        + "PO-1,Vendor A,SKU-1,Item,1,10,each\n"
        + "PO-1,Vendor A,SKU-1,Item,1,10,each\n"
        + "PO-1,Vendor B,SKU-1,Item,1,10,each\n",
    )
    view = _reconciliation_view(result)

    assert view["actionable_exception_count"] == 2
    assert [line["exception_label"] for line in view["exception_lines"]] == [
        "Ambiguous match",
        "Ordered but not invoiced",
    ]
    grouped = [line for line in view["audit_lines"] if line["ambiguity_grouped"]]
    assert len(grouped) == 2
    assert result.summary.missing_on_invoice_count == 3


def test_multiple_upload_rejections_reset_and_session_isolation(tmp_path: Path) -> None:
    app = _app(tmp_path)
    first = app.test_client()
    second = app.test_client()
    response = first.post(
        "/upload",
        data={"documents": [(BytesIO(b"one"), "one.md"), (BytesIO(b"two"), "two.txt")]},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    rejected = second.post(
        "/upload",
        data={"documents": [(BytesIO(b"not supported"), "bad.exe")]},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    reset = first.post("/reset", follow_redirects=True)

    assert b"2 documents" in response.data
    assert b"V1 accepts PDF, DOCX, TXT, Markdown, CSV, and XLSX" in rejected.data
    assert b"one.md" not in second.get("/").data
    assert b"Workspace reset" in reset.data
    assert not app.extensions["knowledge_runs"]


def test_oversized_and_traversal_filenames_fail_closed(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    oversized = _upload(client, "large.md", b"x" * (10 * 1024 * 1024 + 1))
    traversal = _upload(client, "../private.md")

    assert b"Files must be" in oversized.data
    assert b"File names must not include a path" in traversal.data


def test_supported_answer_renders_only_validated_citation_data(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    _upload(client, "policy.md", b"# Attendance\nCall out two hours before a shift.")

    response = client.post(
        "/ask", data={"question": "When should I call out?"}, follow_redirects=True
    )

    assert b"Based on your documents" in response.data
    assert b"Grounded operating guidance." in response.data
    assert b"policy.md" in response.data
    assert b"Call out two hours" in response.data
    assert b"chunk_id" not in response.data


def test_answer_presentation_state_stays_server_side_and_is_session_isolated(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    first = app.test_client()
    second = app.test_client()
    question = "When should I call out?"

    _upload(first, "policy.md", b"# Attendance\nCall out two hours before a shift.")
    response = first.post("/ask", data={"question": question}, follow_redirects=True)

    with first.session_transaction() as browser_session:
        session_data = dict(browser_session)

    assert b"Grounded operating guidance." in response.data
    assert question.encode() in response.data
    assert session_data.keys() == {"run_id"}
    assert isinstance(session_data["run_id"], str)
    assert "answer" not in session_data
    assert "question" not in session_data
    assert "Call out two hours before a shift." not in str(session_data)
    assert b"Grounded operating guidance." not in second.get("/").data

    run = next(iter(app.extensions["knowledge_runs"].values()))
    assert run.question == question
    assert run.answer_view is not None
    assert run.answer_view["citations"][0]["excerpt"] == (
        "Attendance\nCall out two hours before a shift."
    )


def test_harbor_document_and_citation_titles_are_human_readable(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    _upload(
        client,
        "callout_attendance_policy.md",
        b"# Attendance\nCall out two hours before a shift.",
    )

    response = client.post(
        "/ask", data={"question": "When should I call out?"}, follow_redirects=True
    )

    assert b"Call-Out &amp; Attendance Policy" in response.data
    assert b"callout_attendance_policy.md" not in response.data


def test_display_titles_do_not_change_document_identity_or_storage(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    _upload(client, "employee_handbook.md", b"# Handbook\nWelcome.")
    run = next(iter(app.extensions["knowledge_runs"].values()))
    document = run.documents[0]

    assert document_display_title(document.original_display_name) == "Employee Handbook"
    assert document.original_display_name == "employee_handbook.md"
    assert document.stored_filename != document.original_display_name
    assert (run.workspace.uploads_dir / document.stored_filename).is_file()
    display_path = run.workspace.uploads_dir / document_display_title(
        document.original_display_name
    )
    assert not display_path.exists()


def test_client_display_names_are_preserved_when_no_demo_title_exists() -> None:
    assert document_display_title("Q3 Guest Experience Notes.md") == (
        "Q3 Guest Experience Notes.md"
    )


def test_unsupported_and_provider_failure_are_safe_ui_states(tmp_path: Path) -> None:
    unsupported = _app(tmp_path)
    client = unsupported.test_client()
    _upload(client)
    no_support = client.post(
        "/ask",
        data={"question": "What is the CEO home address?"},
        follow_redirects=True,
    )
    failing = _app(tmp_path / "other", FakeAnswerProvider(fail=True)).test_client()
    _upload(failing)
    failure = failing.post("/ask", data={"question": "question"}, follow_redirects=True)

    assert b"I couldn&#39;t find enough support" in no_support.data
    assert b"I couldn&#39;t find enough support" in failure.data
    assert b"private provider output" not in failure.data


def test_empty_workspace_ask_requires_documents(tmp_path: Path) -> None:
    response = (
        _app(tmp_path)
        .test_client()
        .post("/ask", data={"question": "question"}, follow_redirects=True)
    )

    assert b"Add and prepare documents" in response.data
