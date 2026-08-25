"""Small, session-isolated Flask interface for the grounded assistant demo."""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from .answer_generation import generate_grounded_answer
from .chunking import chunk_document
from .extraction import extract_document, read_accepted_content
from .models import (
    AcceptedDocument,
    DocumentType,
    GroundedAnswer,
    ReconciliationIssueCode,
    ReconciliationStatus,
    UploadErrorCode,
)
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
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_DOCUMENTS_PER_RUN = 12
DEFAULT_MAX_TOTAL_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_CHUNKS_PER_RUN = 2_000
DEFAULT_MAX_QUESTION_LENGTH = 2_000
DEFAULT_MAX_ACTIVE_RUNS = 25
DEFAULT_KNOWLEDGE_RUN_TTL_SECONDS = 60 * 60
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
_DEMO_RECONCILIATION_SOURCE_NAMES = frozenset(
    {
        "harbor_hearth_invoices.csv",
        "harbor_hearth_reconciliation_purchase_orders.csv",
    }
)


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
    created_at: float = 0.0
    last_accessed_at: float = 0.0


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create a testable single-process temporary-workspace application."""
    environment = os.environ.get("AI_KNOWLEDGE_ASSISTANT_ENV", "development")
    if environment not in {"development", "production"}:
        raise RuntimeError("Invalid AI_KNOWLEDGE_ASSISTANT_ENV configuration.")
    app = Flask(__name__)
    app.config.from_mapping(
        AI_KNOWLEDGE_ASSISTANT_ENV=environment,
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY"),
        MAX_CONTENT_LENGTH=_env_positive_int(
            "MAX_REQUEST_BYTES", DEFAULT_MAX_TOTAL_UPLOAD_BYTES + 1024 * 1024
        ),
        WORKSPACE_ROOT=(
            Path(os.environ["WORKSPACE_ROOT"])
            if os.environ.get("WORKSPACE_ROOT")
            else None
        ),
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"),
        OPENAI_EMBEDDING_MODEL=os.environ.get("OPENAI_EMBEDDING_MODEL"),
        OPENAI_ANSWER_MODEL=os.environ.get("OPENAI_ANSWER_MODEL"),
        ALLOW_CLIENT_UPLOADS=_env_bool("ALLOW_CLIENT_UPLOADS", True),
        KNOWLEDGE_RUN_TTL_SECONDS=_env_positive_int(
            "KNOWLEDGE_RUN_TTL_SECONDS", DEFAULT_KNOWLEDGE_RUN_TTL_SECONDS
        ),
        MAX_FILE_BYTES=_env_positive_int("MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES),
        MAX_DOCUMENTS_PER_RUN=_env_positive_int(
            "MAX_DOCUMENTS_PER_RUN", DEFAULT_MAX_DOCUMENTS_PER_RUN
        ),
        MAX_TOTAL_UPLOAD_BYTES=_env_positive_int(
            "MAX_TOTAL_UPLOAD_BYTES", DEFAULT_MAX_TOTAL_UPLOAD_BYTES
        ),
        MAX_CHUNKS_PER_RUN=_env_positive_int(
            "MAX_CHUNKS_PER_RUN", DEFAULT_MAX_CHUNKS_PER_RUN
        ),
        MAX_QUESTION_LENGTH=_env_positive_int(
            "MAX_QUESTION_LENGTH", DEFAULT_MAX_QUESTION_LENGTH
        ),
        MAX_ACTIVE_RUNS=_env_positive_int(
            "MAX_ACTIVE_RUNS", DEFAULT_MAX_ACTIVE_RUNS
        ),
        TIME_PROVIDER=time.monotonic,
        CSRF_PROTECTION=True,
        EMBEDDING_PROVIDER=None,
        ANSWER_PROVIDER=None,
    )
    if config:
        app.config.update(config)
    if app.config.get("TESTING") and "CSRF_PROTECTION" not in (config or {}):
        app.config["CSRF_PROTECTION"] = False
    if (
        not app.config["SECRET_KEY"]
        and app.config["AI_KNOWLEDGE_ASSISTANT_ENV"] == "development"
    ):
        app.config["SECRET_KEY"] = secrets.token_urlsafe(32)
    _validate_configuration(app)
    production = app.config["AI_KNOWLEDGE_ASSISTANT_ENV"] == "production"
    app.config.update(
        DEBUG=False,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=production,
    )
    app.extensions["knowledge_runs"] = {}
    app.extensions["knowledge_runs_lock"] = threading.RLock()

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_: RequestEntityTooLarge) -> Any:
        if request.path == "/health":
            return jsonify(status="ok")
        flash("The upload request is too large. Use a smaller upload set.", "error")
        return redirect(url_for("landing"))

    @app.before_request
    def maintain_runs_and_verify_csrf() -> None:
        # Health probes must not traverse or clean a temporary workspace.
        if request.path == "/health":
            return
        _prune_expired_runs(app)
        if request.method == "POST" and app.config["CSRF_PROTECTION"]:
            token = request.form.get("csrf_token")
            if not isinstance(token, str) or not secrets.compare_digest(
                token, _csrf_token()
            ):
                abort(400)
        _touch_current_run(app)

    @app.get("/health")
    def health() -> Any:
        return jsonify(status="ok")

    @app.get("/")
    def landing() -> str:
        return _render(app)

    @app.post("/demo")
    def load_demo() -> Any:
        try:
            run = _reset_run(app)
        except RuntimeError:
            flash(
                "Temporary workspace capacity reached. Please try again shortly.",
                "error",
            )
            return redirect(url_for("landing"))
        try:
            for path in sorted(DEMO_DIRECTORY.iterdir()):
                if not path.is_file():
                    continue
                run.documents.append(
                    accept_upload(run.workspace, path.name, path.read_bytes())
                )
            run.is_demo = True
            _index_run(app, run)
            flash("Synthetic Harbor & Hearth demo is ready.", "success")
        except Exception:
            _discard_run(app, run)
            flash(
                "The demo could not be prepared. Check local provider setup.", "error"
            )
        return redirect(url_for("landing"))

    @app.post("/upload")
    def upload_documents() -> Any:
        if not app.config["ALLOW_CLIENT_UPLOADS"]:
            flash("Client uploads are disabled for this demo.", "error")
            return redirect(url_for("landing"))
        files = tuple(
            file for file in request.files.getlist("documents") if file.filename
        )
        if not files:
            flash("Choose at least one document to upload.", "error")
            return redirect(url_for("landing"))
        if len(files) > app.config["MAX_DOCUMENTS_PER_RUN"]:
            flash("Too many documents. Use a smaller upload set.", "error")
            return redirect(url_for("landing"))
        total_bytes = sum(_uploaded_file_size(file) for file in files)
        if total_bytes > app.config["MAX_TOTAL_UPLOAD_BYTES"]:
            flash("The total upload is too large. Use a smaller upload set.", "error")
            return redirect(url_for("landing"))
        try:
            run = _reset_run(app)
        except RuntimeError:
            flash(
                "Temporary workspace capacity reached. Please try again shortly.",
                "error",
            )
            return redirect(url_for("landing"))
        try:
            for file in files:
                run.documents.append(
                    accept_upload(
                        run.workspace,
                        file.filename,
                        file.read(),
                        max_file_size_bytes=app.config["MAX_FILE_BYTES"],
                    )
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
        if len(question) > app.config["MAX_QUESTION_LENGTH"]:
            flash(
                "Questions must be shorter. Please ask a more concise question.",
                "error",
            )
            return redirect(url_for("landing"))
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
        csrf_token=_csrf_token(),
        allow_client_uploads=app.config["ALLOW_CLIENT_UPLOADS"],
    )


def _current_run(app: Flask) -> KnowledgeRun | None:
    run_id = session.get("run_id")
    if not isinstance(run_id, str):
        return None
    with app.extensions["knowledge_runs_lock"]:
        return app.extensions["knowledge_runs"].get(run_id)


def _reset_run(app: Flask) -> KnowledgeRun:
    current = _current_run(app)
    if current is not None:
        _discard_run(app, current)
    with app.extensions["knowledge_runs_lock"]:
        runs = app.extensions["knowledge_runs"]
        if len(runs) >= app.config["MAX_ACTIVE_RUNS"]:
            raise RuntimeError("Temporary workspace capacity reached.")
        workspace = UploadWorkspace.create(app.config["WORKSPACE_ROOT"])
        now = app.config["TIME_PROVIDER"]()
        run = KnowledgeRun(workspace=workspace, created_at=now, last_accessed_at=now)
        runs[workspace.run_id] = run
    session["run_id"] = workspace.run_id
    # Clear only legacy presentation values that an older browser cookie could
    # contain; all current presentation state belongs to the server-side run.
    session.pop("answer", None)
    session.pop("question", None)
    return run


def _discard_run(app: Flask, run: KnowledgeRun) -> None:
    with app.extensions["knowledge_runs_lock"]:
        app.extensions["knowledge_runs"].pop(run.workspace.run_id, None)
    try:
        run.workspace.cleanup()
    except Exception:
        pass


def _providers(app: Flask) -> tuple[EmbeddingProvider, Any]:
    embedding = app.config["EMBEDDING_PROVIDER"] or OpenAIEmbeddingProvider(
        api_key=app.config["OPENAI_API_KEY"],
        model=app.config["OPENAI_EMBEDDING_MODEL"],
    )
    answer = app.config["ANSWER_PROVIDER"] or OpenAIAnswerProvider(
        api_key=app.config["OPENAI_API_KEY"], model=app.config["OPENAI_ANSWER_MODEL"]
    )
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
    reconciliation_records = (
        tuple(
            record
            for record in records
            if record.document_name in _DEMO_RECONCILIATION_SOURCE_NAMES
        )
        if run.is_demo
        else records
    )
    run.reconciliation = reconcile(reconciliation_records)
    if run.reconciliation.lines:
        extracted.append(reconciliation_evidence(run.reconciliation))
    chunks = tuple(
        chunk for document in extracted for chunk in chunk_document(document)
    )
    if len(chunks) > app.config["MAX_CHUNKS_PER_RUN"]:
        raise UploadValidationError(
            UploadErrorCode.STRUCTURED_LIMIT_EXCEEDED,
            "Documents create too much content. Use a smaller upload set.",
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
    engine_exception_lines = [
        line
        for line in result.lines
        if line.status is not ReconciliationStatus.MATCHED or line.issue_codes
    ]

    # An ambiguous invoice line is one operator decision.  The engine also
    # preserves its unconsumed PO candidates as MISSING_ON_INVOICE lines; keep
    # those exact lines in the audit projection but do not promote each one to
    # a separate default business problem.
    grouped_candidate_ids = {
        candidate_id
        for line in engine_exception_lines
        if ReconciliationIssueCode.AMBIGUOUS_MATCH in line.issue_codes
        for candidate_id in line.ambiguity_candidate_po_record_ids
    }

    default_exception_lines = [
        line
        for line in engine_exception_lines
        if not (
            line.status is ReconciliationStatus.MISSING_ON_INVOICE
            and line.purchase_order
            and line.purchase_order.record_id in grouped_candidate_ids
        )
    ]

    def display_line(line: Any) -> dict[str, Any]:
        quantity = line.quantity_variance
        price = line.unit_price_variance
        extended = line.extended_variance
        return {
            "status": line.status.value.replace("_", " ").title(),
            "exception_label": _exception_label(line),
            "invoice": line.invoice_number or "—",
            "po": line.po_number or "—",
            "item": line.item_name or "Unidentified line",
            "issues": ", ".join(
                code.value.replace("_", " ").title() for code in line.issue_codes
            ),
            "quantity_ordered": _decimal_text(quantity.po_quantity)
            if quantity
            else None,
            "quantity_invoiced": _decimal_text(quantity.invoice_quantity)
            if quantity
            else None,
            "quantity_difference": _signed_decimal_text(quantity.variance)
            if quantity
            else None,
            "quantity_unit": line.po_unit or line.invoice_unit,
            "po_unit_price": _money_text(price.po_amount) if price else None,
            "invoice_unit_price": _money_text(price.invoice_amount) if price else None,
            "unit_price_difference": _signed_money_text(price.variance)
            if price
            else None,
            "extended_variance": _signed_money_text(extended.variance)
            if extended
            else None,
            "invoice_unit": line.invoice_unit,
            "po_unit": line.po_unit,
            "ambiguity_candidate_count": len(line.ambiguity_candidate_po_record_ids)
            if ReconciliationIssueCode.AMBIGUOUS_MATCH in line.issue_codes
            else None,
            "ambiguity_grouped": bool(
                line.purchase_order
                and line.purchase_order.record_id in grouped_candidate_ids
            ),
        }

    priority = {
        ReconciliationStatus.VARIANCE: 0,
        ReconciliationStatus.MISSING_ON_PO: 1,
        ReconciliationStatus.UNMATCHED: 2,
        ReconciliationStatus.MISSING_ON_INVOICE: 3,
        ReconciliationStatus.MATCHED: 4,
    }
    ordered_exceptions = sorted(
        default_exception_lines, key=lambda line: priority[line.status]
    )
    return {
        "matched": summary.matched_line_count,
        "variances": summary.variance_line_count,
        "missing_on_po": summary.missing_on_po_count,
        "missing_on_invoice": summary.missing_on_invoice_count,
        "total_variance": f"${summary.total_monetary_variance:.2f}",
        "exception_count": len(default_exception_lines),
        "actionable_exception_count": len(default_exception_lines),
        "total_line_count": len(result.lines),
        "exception_lines": tuple(display_line(line) for line in ordered_exceptions),
        "audit_lines": tuple(display_line(line) for line in result.lines),
    }


def _decimal_text(value: Any) -> str:
    rendered = format(value, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _signed_decimal_text(value: Any) -> str:
    return ("+" if value >= 0 else "-") + _decimal_text(abs(value))


def _money_text(value: Any) -> str:
    return f"${value:.2f}"


def _signed_money_text(value: Any) -> str:
    return ("+$" if value >= 0 else "-$") + f"{abs(value):.2f}"


def _exception_label(line: Any) -> str:
    """Translate deterministic outcomes into concise client-facing language."""
    issues = set(line.issue_codes)
    labels = []
    if line.status is ReconciliationStatus.MATCHED and not issues:
        labels.append("Matched")
    elif line.status is ReconciliationStatus.VARIANCE:
        has_price = bool(line.unit_price_variance and line.unit_price_variance.variance)
        has_quantity = bool(line.quantity_variance and line.quantity_variance.variance)
        if has_price and has_quantity:
            labels.append("Quantity & price variance")
        elif has_price:
            labels.append("Price variance")
        elif has_quantity:
            labels.append("Quantity variance")
        if not labels:
            labels.append("Line total variance")
    elif line.status is ReconciliationStatus.MISSING_ON_PO:
        labels.append("Not found on purchase order")
    elif line.status is ReconciliationStatus.MISSING_ON_INVOICE:
        labels.append("Ordered but not invoiced")
    if ReconciliationIssueCode.UNIT_MISMATCH in issues:
        labels.append("Unit mismatch")
    if ReconciliationIssueCode.AMBIGUOUS_MATCH in issues:
        labels.append("Ambiguous match")
    issue_labels = {
        ReconciliationIssueCode.INVALID_QUANTITY: "Invalid quantity",
        ReconciliationIssueCode.INVALID_UNIT_PRICE: "Invalid unit price",
        ReconciliationIssueCode.INVALID_LINE_TOTAL: "Invalid line total",
        ReconciliationIssueCode.MISSING_PO_NUMBER: "Missing purchase-order number",
        ReconciliationIssueCode.PO_NOT_FOUND: "Purchase order not found",
        ReconciliationIssueCode.ITEM_NOT_FOUND: "Item not found",
        ReconciliationIssueCode.UNSUPPORTED_SCHEMA: "Unsupported record format",
        ReconciliationIssueCode.WRONG_RECORD_TYPE: "Unsupported record type",
    }
    labels.extend(
        issue_labels[issue] for issue in line.issue_codes if issue in issue_labels
    )
    return " · ".join(dict.fromkeys(labels)) or "Needs review"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise RuntimeError(f"Invalid {name} configuration.")


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise RuntimeError(f"Invalid {name} configuration.") from None
    if parsed <= 0:
        raise RuntimeError(f"Invalid {name} configuration.")
    return parsed


def _validate_configuration(app: Flask) -> None:
    if app.config["AI_KNOWLEDGE_ASSISTANT_ENV"] not in {"development", "production"}:
        raise RuntimeError("Invalid AI_KNOWLEDGE_ASSISTANT_ENV configuration.")
    if app.config["AI_KNOWLEDGE_ASSISTANT_ENV"] == "production" and not os.environ.get(
        "FLASK_SECRET_KEY"
    ) and not app.config.get("SECRET_KEY"):
        raise RuntimeError("Production requires FLASK_SECRET_KEY configuration.")
    for key in (
        "MAX_CONTENT_LENGTH",
        "KNOWLEDGE_RUN_TTL_SECONDS",
        "MAX_FILE_BYTES",
        "MAX_DOCUMENTS_PER_RUN",
        "MAX_TOTAL_UPLOAD_BYTES",
        "MAX_CHUNKS_PER_RUN",
        "MAX_QUESTION_LENGTH",
        "MAX_ACTIVE_RUNS",
    ):
        if not isinstance(app.config[key], int) or app.config[key] <= 0:
            raise RuntimeError(f"Invalid {key} configuration.")
    if app.config["MAX_CONTENT_LENGTH"] < app.config["MAX_TOTAL_UPLOAD_BYTES"]:
        raise RuntimeError("MAX_REQUEST_BYTES must allow the total upload limit.")


def _csrf_token() -> str:
    if not current_app.config["CSRF_PROTECTION"]:
        return ""
    token = session.get("csrf_token")
    if not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _uploaded_file_size(file: Any) -> int:
    stream = file.stream
    position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(position)
    return size


def _touch_current_run(app: Flask) -> None:
    run = _current_run(app)
    if run is not None:
        with app.extensions["knowledge_runs_lock"]:
            if app.extensions["knowledge_runs"].get(run.workspace.run_id) is run:
                run.last_accessed_at = app.config["TIME_PROVIDER"]()


def _prune_expired_runs(app: Flask) -> None:
    now = app.config["TIME_PROVIDER"]()
    with app.extensions["knowledge_runs_lock"]:
        expired = tuple(
            run for run in app.extensions["knowledge_runs"].values()
            if now - run.last_accessed_at >= app.config["KNOWLEDGE_RUN_TTL_SECONDS"]
        )
    for run in expired:
        _discard_run(app, run)


if __name__ == "__main__":  # pragma: no cover
    create_app().run(
        host=os.environ.get("AI_KNOWLEDGE_ASSISTANT_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
