"""Small, session-isolated Flask interface for the grounded assistant demo."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, session, url_for

from .answer_generation import generate_grounded_answer
from .chunking import chunk_document
from .extraction import extract_document, read_accepted_content
from .models import AcceptedDocument, DocumentType, GroundedAnswer
from .openai_answers import OpenAIAnswerProvider
from .openai_embeddings import OpenAIEmbeddingProvider
from .reconciliation import reconcile, reconciliation_evidence
from .retrieval import EmbeddingProvider, LocalVectorIndex, build_index, retrieve
from .structured_records import parse_structured_document, structured_evidence
from .uploads import UploadValidationError, accept_upload
from .workspace import UploadWorkspace

DEMO_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "demo_documents" / "harbor_and_hearth"
)
MAX_REQUEST_BYTES = 10 * 1024 * 1024 * 8
_DEMO_DISPLAY_TITLES = {
    "callout_attendance_policy.md": "Call-Out & Attendance Policy",
    "employee_handbook.md": "Employee Handbook",
    "menu_product_reference.md": "Menu & Product Reference Guide",
    "new_team_member_training_guide.md": "New Team Member Training Guide",
    "opening_closing_sop.md": "Opening & Closing SOP",
    "refund_service_recovery_policy.md": "Refund & Service Recovery Policy",
    "harbor_hearth_invoices.csv": "Harbor & Hearth Invoices (fictional)",
    "harbor_hearth_reconciliation_purchase_orders.csv": (
        "Harbor & Hearth Reconciliation Purchase Orders (fictional)"
    ),
    "harbor_hearth_vendors.csv": "Harbor & Hearth Vendors (fictional)",
    "harbor_hearth_purchase_orders.xlsx": "Harbor & Hearth Purchase Orders (fictional)",
    "harbor_hearth_products.xlsx": "Harbor & Hearth Products (fictional)",
}


@dataclass(slots=True)
class KnowledgeRun:
    """Server-only state for one temporary browser workspace."""

    workspace: UploadWorkspace
    documents: list[AcceptedDocument] = field(default_factory=list)
    index: LocalVectorIndex | None = None
    chunk_count: int = 0
    is_demo: bool = False
    question: str | None = None
    answer_view: dict[str, Any] | None = None
    reconciliation: Any | None = None


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create a testable, state-minimal local Flask application."""
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(32),
        MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
        WORKSPACE_ROOT=None,
        EMBEDDING_PROVIDER=None,
        ANSWER_PROVIDER=None,
    )
    if config:
        app.config.update(config)
    app.extensions["knowledge_runs"] = {}

    @app.get("/")
    def landing() -> str:
        return _render(app)

    @app.post("/demo")
    def load_demo() -> Any:
        run = _reset_run(app)
        try:
            for path in sorted(DEMO_DIRECTORY.iterdir()):
                if not path.is_file():
                    continue
                run.documents.append(
                    accept_upload(run.workspace, path.name, path.read_bytes())
                )
            _index_run(app, run)
            run.is_demo = True
            flash("Synthetic Harbor & Hearth demo is ready.", "success")
        except Exception:
            _discard_run(app, run)
            flash(
                "The demo could not be prepared. Check local provider setup.", "error"
            )
        return redirect(url_for("landing"))

    @app.post("/upload")
    def upload_documents() -> Any:
        files = tuple(
            file for file in request.files.getlist("documents") if file.filename
        )
        if not files:
            flash("Choose at least one document to upload.", "error")
            return redirect(url_for("landing"))
        run = _reset_run(app)
        try:
            for file in files:
                run.documents.append(
                    accept_upload(run.workspace, file.filename, file.read())
                )
            _index_run(app, run)
            flash("Documents are indexed and ready for grounded questions.", "success")
        except UploadValidationError as error:
            _discard_run(app, run)
            flash(error.message, "error")
        except Exception:
            _discard_run(app, run)
            flash(
                "Documents could not be prepared. Check local provider setup.", "error"
            )
        return redirect(url_for("landing"))

    @app.post("/ask")
    def ask() -> Any:
        question = request.form.get("question", "")
        run = _current_run(app)
        if run is None or run.index is None:
            flash("Add and prepare documents before asking a question.", "error")
            return redirect(url_for("landing"))
        try:
            embedding_provider, answer_provider = _providers(app)
            result = retrieve(run.index, question, embedding_provider)
            answer = generate_grounded_answer(question, result.sources, answer_provider)
            run.question = result.question
            run.answer_view = _answer_view(answer)
        except Exception:
            flash(
                "The question could not be answered. Check local provider setup.",
                "error",
            )
        return redirect(url_for("landing") + "#answer")

    @app.post("/reset")
    def reset_workspace() -> Any:
        previous = _current_run(app)
        if previous is not None:
            _discard_run(app, previous)
        session.pop("run_id", None)
        session.pop("answer", None)
        session.pop("question", None)
        flash("Workspace reset. Temporary documents were removed.", "success")
        return redirect(url_for("landing"))

    return app


def _render(app: Flask) -> str:
    run = _current_run(app)
    return render_template(
        "index.html",
        run=run,
        answer=run.answer_view if run else None,
        question=run.question if run else None,
        examples=_examples() if run and run.is_demo else (),
        reconciliation=_reconciliation_view(run.reconciliation)
        if run and run.reconciliation
        else None,
        document_display_title=document_display_title,
    )


def _current_run(app: Flask) -> KnowledgeRun | None:
    run_id = session.get("run_id")
    if not isinstance(run_id, str):
        return None
    return app.extensions["knowledge_runs"].get(run_id)


def _reset_run(app: Flask) -> KnowledgeRun:
    current = _current_run(app)
    if current is not None:
        _discard_run(app, current)
    workspace = UploadWorkspace.create(app.config["WORKSPACE_ROOT"])
    run = KnowledgeRun(workspace=workspace)
    app.extensions["knowledge_runs"][workspace.run_id] = run
    session["run_id"] = workspace.run_id
    # Clear only legacy presentation values that an older browser cookie could
    # contain; all current presentation state belongs to the server-side run.
    session.pop("answer", None)
    session.pop("question", None)
    return run


def _discard_run(app: Flask, run: KnowledgeRun) -> None:
    app.extensions["knowledge_runs"].pop(run.workspace.run_id, None)
    try:
        run.workspace.cleanup()
    except Exception:
        pass


def _providers(app: Flask) -> tuple[EmbeddingProvider, Any]:
    embedding = app.config["EMBEDDING_PROVIDER"] or OpenAIEmbeddingProvider()
    answer = app.config["ANSWER_PROVIDER"] or OpenAIAnswerProvider()
    return embedding, answer


def _index_run(app: Flask, run: KnowledgeRun) -> None:
    embedding_provider, _ = _providers(app)
    structured, extracted = [], []
    for document in run.documents:
        if document.document_type in {DocumentType.CSV, DocumentType.XLSX}:
            parsed = parse_structured_document(
                document, read_accepted_content(run.workspace, document)
            )
            structured.append(parsed)
            extracted.append(structured_evidence(parsed))
        else:
            extracted.append(extract_document(run.workspace, document))
    records = tuple(
        record
        for document in structured
        for sheet in document.sheets
        for record in sheet.records
    )
    run.reconciliation = reconcile(records)
    if run.reconciliation.lines:
        extracted.append(reconciliation_evidence(run.reconciliation))
    chunks = tuple(
        chunk for document in extracted for chunk in chunk_document(document)
    )
    run.index = build_index(chunks, embedding_provider)
    run.chunk_count = len(chunks)


def _answer_view(answer: GroundedAnswer) -> dict[str, Any]:
    return {
        "status": answer.status.value,
        "text": answer.answer,
        "citations": tuple(
            {
                "document_name": citation.document_name,
                "display_title": document_display_title(citation.document_name),
                "locator": _locator_text(citation),
                "excerpt": citation.source_excerpt,
                "rank": citation.rank,
            }
            for citation in answer.citations
        ),
    }


def document_display_title(document_name: str) -> str:
    """Return a presentation-only title without changing document identity.

    Harbor & Hearth's committed fixtures use intentional business titles. Other
    accepted uploads retain their validated, user-supplied display names so a
    client filename is not guessed at, normalized into a path, or used for
    storage.
    """
    return _DEMO_DISPLAY_TITLES.get(document_name, document_name)


def _locator_text(citation: Any) -> str:
    locator = citation.source_locator
    if locator.page_number:
        return f"Page {locator.page_number}"
    if locator.section_label:
        return locator.section_label
    if locator.line_start:
        return f"Lines {locator.line_start}–{locator.line_end or locator.line_start}"
    if locator.row_number:
        base = (
            f"{locator.sheet_name or citation.document_name} — Row {locator.row_number}"
        )
        return f"{locator.record_label} — {base}" if locator.record_label else base
    return "Document section"


def _examples() -> tuple[str, ...]:
    return (
        "How far in advance should I call out for a shift?",
        "What are the opening cash-drawer steps?",
        "What should I do if a guest requests a refund because of an allergen concern?",
        "What is the CEO’s home address?",
        "Did invoice INV-1048 match its purchase order?",
        "Which invoice lines differ from their purchase orders?",
    )


def _reconciliation_view(result: Any) -> dict[str, Any]:
    """Safe display projection; never expose internal IDs or source paths."""
    summary = result.summary
    return {
        "matched": summary.matched_line_count,
        "variances": summary.variance_line_count,
        "missing_on_po": summary.missing_on_po_count,
        "missing_on_invoice": summary.missing_on_invoice_count,
        "total_variance": f"${summary.total_monetary_variance:.2f}",
        "lines": tuple(
            {
                "status": line.status.value.replace("_", " ").title(),
                "invoice": line.invoice_number or "—",
                "po": line.po_number or "—",
                "item": line.item_name or "Unidentified line",
                "issues": ", ".join(
                    code.value.replace("_", " ").title() for code in line.issue_codes
                ),
                "variance": f"${line.extended_variance.variance:+.2f}"
                if line.extended_variance
                else "—",
            }
            for line in result.lines
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    create_app().run(
        host=os.environ.get("AI_KNOWLEDGE_ASSISTANT_HOST", "127.0.0.1"),
        port=5000,
        debug=False,
    )
