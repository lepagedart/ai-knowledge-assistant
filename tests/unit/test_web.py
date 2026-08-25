"""Offline Flask test-client coverage for the local knowledge-assistant UI."""

from __future__ import annotations

import importlib
import re
import sys
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Sequence

import pytest

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
    _prune_expired_runs,
    _reconciliation_view,
    _reset_run,
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


def test_health_is_offline_and_does_not_construct_a_provider(tmp_path: Path) -> None:
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret", "WORKSPACE_ROOT": tmp_path}
    )

    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_wsgi_startup_is_offline_with_valid_production_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AI_KNOWLEDGE_ASSISTANT_ENV", "production")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-production-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("ai_knowledge_assistant.wsgi", None)

    wsgi = importlib.import_module("ai_knowledge_assistant.wsgi")

    assert wsgi.app.config["DEBUG"] is False
    assert wsgi.app.test_client().get("/health").json == {"status": "ok"}


def test_wsgi_production_startup_requires_a_secret_key(monkeypatch) -> None:
    monkeypatch.setenv("AI_KNOWLEDGE_ASSISTANT_ENV", "production")
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("ai_knowledge_assistant.wsgi", None)

    with pytest.raises(
        RuntimeError, match="Production requires FLASK_SECRET_KEY configuration"
    ):
        importlib.import_module("ai_knowledge_assistant.wsgi")


def test_production_requires_secret_key_without_leaking_a_value(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    try:
        create_app(
            {
                "AI_KNOWLEDGE_ASSISTANT_ENV": "production",
                "OPENAI_API_KEY": "not-a-real-key",
                "WORKSPACE_ROOT": tmp_path,
            }
        )
    except RuntimeError as error:
        assert str(error) == "Production requires FLASK_SECRET_KEY configuration."
        assert "not-a-real-key" not in str(error)
    else:  # pragma: no cover
        raise AssertionError("Production configuration unexpectedly started.")


def test_malformed_environment_configuration_fails_without_secret_disclosure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FLASK_SECRET_KEY", "private-test-secret")
    monkeypatch.setenv("ALLOW_CLIENT_UPLOADS", "perhaps")

    with pytest.raises(RuntimeError, match="Invalid ALLOW_CLIENT_UPLOADS") as error:
        create_app({"WORKSPACE_ROOT": tmp_path})

    assert "private-test-secret" not in str(error.value)


def test_non_positive_ttl_configuration_is_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KNOWLEDGE_RUN_TTL_SECONDS", "0")

    with pytest.raises(
        RuntimeError, match="Invalid KNOWLEDGE_RUN_TTL_SECONDS"
    ):
        create_app({"WORKSPACE_ROOT": tmp_path})


def test_production_cookie_flags_and_development_http_behavior(tmp_path: Path) -> None:
    production = create_app(
        {
            "AI_KNOWLEDGE_ASSISTANT_ENV": "production",
            "SECRET_KEY": "test-secret",
            "EMBEDDING_PROVIDER": FakeEmbeddingProvider(),
            "ANSWER_PROVIDER": FakeAnswerProvider(),
            "WORKSPACE_ROOT": tmp_path / "production",
        }
    )
    development = _app(tmp_path / "development")

    assert production.config["DEBUG"] is False
    assert production.config["SESSION_COOKIE_SECURE"] is True
    assert production.config["SESSION_COOKIE_HTTPONLY"] is True
    assert production.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert development.config["SESSION_COOKIE_SECURE"] is False


def test_expired_runs_are_removed_but_active_runs_are_preserved(tmp_path: Path) -> None:
    clock = [100.0]
    app = _app(tmp_path)
    app.config.update(TIME_PROVIDER=lambda: clock[0], KNOWLEDGE_RUN_TTL_SECONDS=60)
    with app.test_request_context("/"):
        expired = _reset_run(app)
    clock[0] = 150.0
    with app.test_request_context("/"):
        active = _reset_run(app)
    clock[0] = 161.0
    _prune_expired_runs(app)

    runs = app.extensions["knowledge_runs"]
    assert expired.workspace.run_id not in runs
    assert not expired.workspace.root.exists()
    assert active.workspace.run_id in runs
    assert active.workspace.root.exists()


def test_ttl_cleanup_honors_exact_and_adjacent_boundaries(tmp_path: Path) -> None:
    clock = [100.0]
    app = _app(tmp_path)
    app.config.update(TIME_PROVIDER=lambda: clock[0], KNOWLEDGE_RUN_TTL_SECONDS=60)
    with app.test_request_context("/"):
        run = _reset_run(app)

    clock[0] = 159.999
    _prune_expired_runs(app)
    assert run.workspace.run_id in app.extensions["knowledge_runs"]

    clock[0] = 160.0
    _prune_expired_runs(app)
    assert run.workspace.run_id not in app.extensions["knowledge_runs"]
    assert not run.workspace.root.exists()


def test_health_does_not_prune_or_refresh_a_run(tmp_path: Path) -> None:
    clock = [100.0]
    app = _app(tmp_path)
    app.config.update(TIME_PROVIDER=lambda: clock[0], KNOWLEDGE_RUN_TTL_SECONDS=60)
    with app.test_request_context("/"):
        run = _reset_run(app)
    clock[0] = 200.0

    response = app.test_client().get("/health")

    assert response.json == {"status": "ok"}
    assert app.extensions["knowledge_runs"][run.workspace.run_id] is run
    assert run.last_accessed_at == 100.0


def test_uploads_can_be_disabled_server_side_while_demo_still_works(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    app.config["ALLOW_CLIENT_UPLOADS"] = False
    client = app.test_client()

    blocked = _upload(client)
    demo = client.post("/demo", follow_redirects=True)

    assert b"Client uploads are disabled" in blocked.data
    assert b"Synthetic Harbor &amp; Hearth demo" in demo.data
    assert b"name=\"documents\"" not in demo.data


def test_capacity_limits_reject_oversized_document_sets_and_questions(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    app.config.update(MAX_DOCUMENTS_PER_RUN=1, MAX_QUESTION_LENGTH=5)
    client = app.test_client()

    too_many = client.post(
        "/upload",
        data={
            "documents": [
                (BytesIO(b"# one"), "one.md"),
                (BytesIO(b"# two"), "two.md"),
            ]
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    _upload(client)
    question = client.post("/ask", data={"question": "too long"}, follow_redirects=True)

    assert b"Too many documents" in too_many.data
    assert b"Questions must be shorter" in question.data


def test_total_upload_capacity_limit_rejects_request(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.config["MAX_TOTAL_UPLOAD_BYTES"] = 4

    response = _upload(app.test_client(), content=b"too large")

    assert b"total upload is too large" in response.data


def test_oversized_request_has_a_sanitized_browser_response(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.config["MAX_CONTENT_LENGTH"] = 100

    response = _upload(app.test_client(), content=b"x" * 1_000)

    assert b"upload request is too large" in response.data
    assert b"RequestEntityTooLarge" not in response.data


def test_csrf_protects_public_post_routes_and_allows_valid_form(tmp_path: Path) -> None:
    app = create_app(
        {
            "SECRET_KEY": "test-secret",
            "WORKSPACE_ROOT": tmp_path,
            "EMBEDDING_PROVIDER": FakeEmbeddingProvider(),
            "ANSWER_PROVIDER": FakeAnswerProvider(),
        }
    )
    client = app.test_client()

    assert client.post("/demo").status_code == 400
    landing = client.get("/")
    token = re.search(rb'name="csrf_token" value="([^"]+)"', landing.data)
    assert token is not None
    response = client.post("/demo", data={"csrf_token": token.group(1).decode()})

    assert response.status_code == 302
    assert len(app.extensions["knowledge_runs"]) == 1


def test_csrf_is_required_for_every_public_post_route(tmp_path: Path) -> None:
    app = create_app({"SECRET_KEY": "test-secret", "WORKSPACE_ROOT": tmp_path})
    client = app.test_client()

    for path in ("/demo", "/upload", "/ask", "/reset"):
        assert client.post(path).status_code == 400


def test_csrf_token_from_another_browser_session_is_rejected(tmp_path: Path) -> None:
    app = create_app(
        {
            "SECRET_KEY": "test-secret",
            "WORKSPACE_ROOT": tmp_path,
            "EMBEDDING_PROVIDER": FakeEmbeddingProvider(),
            "ANSWER_PROVIDER": FakeAnswerProvider(),
        }
    )
    first = app.test_client()
    second = app.test_client()
    token = re.search(rb'name="csrf_token" value="([^"]+)"', first.get("/").data)
    assert token is not None

    response = second.post("/demo", data={"csrf_token": token.group(1).decode()})

    assert response.status_code == 400


def test_active_run_limit_rejects_new_runs_without_deleting_existing_ones(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    app.config["MAX_ACTIVE_RUNS"] = 1
    first = app.test_client()
    second = app.test_client()

    first_response = first.post("/demo", follow_redirects=True)
    existing = next(iter(app.extensions["knowledge_runs"].values()))
    second_response = second.post("/demo", follow_redirects=True)

    assert b"Synthetic Harbor &amp; Hearth demo is ready" in first_response.data
    assert b"Temporary workspace capacity reached" in second_response.data
    assert tuple(app.extensions["knowledge_runs"].values()) == (existing,)
    assert existing.workspace.root.exists()


def test_browser_sessions_remain_isolated_when_one_resets(tmp_path: Path) -> None:
    app = _app(tmp_path)
    first = app.test_client()
    second = app.test_client()

    _upload(first, name="first.md", content=b"# First\nAlpha only")
    _upload(second, name="second.md", content=b"# Second\nBeta only")
    with first.session_transaction() as first_session:
        first_run_id = first_session["run_id"]
    with second.session_transaction() as second_session:
        second_run_id = second_session["run_id"]
    second_run = app.extensions["knowledge_runs"][second_run_id]

    reset = first.post("/reset", follow_redirects=True)

    assert first_run_id not in app.extensions["knowledge_runs"]
    assert second_run_id in app.extensions["knowledge_runs"]
    assert second_run.workspace.root.exists()
    assert [document.original_display_name for document in second_run.documents] == [
        "second.md"
    ]
    assert b"Workspace reset" in reset.data


def test_deployment_documentation_sets_the_single_worker_policy() -> None:
    deployment = Path("docs/portfolio-deployment.md").read_text()
    render = Path("render.yaml").read_text()

    assert "exactly one Gunicorn worker and one\nthread" in deployment
    assert "--workers 1 --threads 1" in render


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
