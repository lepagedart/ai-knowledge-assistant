"""Offline, deterministic invoice-to-purchase-order reconciliation.

Matching never uses embeddings, fuzzy matching, or an answer provider.  A PO
number narrows the document context; within it, SKU is preferred over an exact
normalized item name and vendor context is used as an additional exact filter.
Ambiguous candidates are deliberately left unresolved.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from .models import (
    DocumentType,
    ExtractedDocument,
    ExtractedSection,
    MatchedRecordReference,
    MoneyVariance,
    QuantityVariance,
    ReconciliationIssueCode,
    ReconciliationLine,
    ReconciliationResult,
    ReconciliationStatus,
    ReconciliationSummary,
    SourceLocator,
    SourceLocatorKind,
    StructuredRecord,
    StructuredRecordType,
)

RECONCILIATION_VERSION = "reconciliation-v1"
DEFAULT_MONETARY_TOLERANCE = Decimal("0.01")
RECONCILIATION_DOCUMENT_NAME = "Reconciliation Results (locally computed)"


def reconcile(
    records: tuple[StructuredRecord, ...],
    *,
    monetary_tolerance: Decimal = DEFAULT_MONETARY_TOLERANCE,
) -> ReconciliationResult:
    """Compare valid invoice and PO rows without changing source values.

    A price/total difference whose absolute value is at most the tolerance is
    retained in the line but does not make the line a variance. Quantities are
    always exact and units are never converted. Each PO row is consumed at
    most once in stable invoice source order.
    """
    if monetary_tolerance < 0:
        raise ValueError("Monetary tolerance must not be negative.")
    invoices = [r for r in records if r.record_type is StructuredRecordType.INVOICE]
    pos = [r for r in records if r.record_type is StructuredRecordType.PURCHASE_ORDER]
    po_by_number: dict[str, list[StructuredRecord]] = defaultdict(list)
    for po in pos:
        number = _value(po, "po_number")
        if number:
            po_by_number[_key(number)].append(po)
    consumed: set[str] = set()
    lines: list[ReconciliationLine] = []
    # ``records`` retains parser/source order, which is the deterministic
    # consumption order when identical invoice lines compete for one PO line.
    for invoice in invoices:
        po_number = _value(invoice, "po_number")
        if not po_number:
            lines.append(
                _line(
                    invoice,
                    None,
                    ReconciliationStatus.UNMATCHED,
                    (ReconciliationIssueCode.MISSING_PO_NUMBER,),
                )
            )
            continue
        candidates = po_by_number.get(_key(po_number), [])
        if not candidates:
            lines.append(
                _line(
                    invoice,
                    None,
                    ReconciliationStatus.UNMATCHED,
                    (ReconciliationIssueCode.PO_NOT_FOUND,),
                )
            )
            continue
        available_candidates = [
            candidate for candidate in candidates if candidate.record_id not in consumed
        ]
        matches, issue = _matching_candidates(invoice, available_candidates)
        if len(matches) != 1:
            code = issue or ReconciliationIssueCode.ITEM_NOT_FOUND
            status = (
                ReconciliationStatus.MISSING_ON_PO
                if code is ReconciliationIssueCode.ITEM_NOT_FOUND
                else ReconciliationStatus.UNMATCHED
            )
            lines.append(
                _line(
                    invoice,
                    None,
                    status,
                    (code,),
                    ambiguity_candidate_po_record_ids=tuple(
                        candidate.record_id for candidate in matches
                    )
                    if code is ReconciliationIssueCode.AMBIGUOUS_MATCH
                    else (),
                )
            )
            continue
        po = matches[0]
        consumed.add(po.record_id)
        lines.append(_compare(invoice, po, monetary_tolerance))
    for po in pos:
        if po.record_id not in consumed:
            lines.append(_line(None, po, ReconciliationStatus.MISSING_ON_INVOICE, ()))
    ordered = tuple(sorted(lines, key=lambda line: line.reconciliation_id))
    money_lines = [
        line.extended_variance
        for line in ordered
        if line.status in {ReconciliationStatus.MATCHED, ReconciliationStatus.VARIANCE}
        and line.extended_variance
    ]
    summary = ReconciliationSummary(
        sum(line.status is ReconciliationStatus.MATCHED for line in ordered),
        sum(line.status is ReconciliationStatus.VARIANCE for line in ordered),
        sum(line.status is ReconciliationStatus.MISSING_ON_PO for line in ordered),
        sum(line.status is ReconciliationStatus.MISSING_ON_INVOICE for line in ordered),
        sum((variance.variance for variance in money_lines), Decimal("0")),
        len(money_lines),
    )
    return ReconciliationResult(RECONCILIATION_VERSION, ordered, summary)


def reconciliation_evidence(result: ReconciliationResult) -> ExtractedDocument:
    """Render local calculations as citation-ready evidence, without AI text."""
    document_id = hashlib.sha256(
        (
            RECONCILIATION_VERSION
            + "|"
            + "|".join(line.reconciliation_id for line in result.lines)
        ).encode()
    ).hexdigest()
    sections = tuple(
        ExtractedSection(
            line.reconciliation_id,
            document_id,
            RECONCILIATION_DOCUMENT_NAME,
            DocumentType.TEXT,
            SourceLocator(
                SourceLocatorKind.DOCUMENT_SECTION, section_label=_label(line)
            ),
            render_reconciliation_evidence(line),
            hashlib.sha256(render_reconciliation_evidence(line).encode()).hexdigest(),
        )
        for line in result.lines
    )
    return ExtractedDocument(
        RECONCILIATION_VERSION,
        document_id,
        RECONCILIATION_DOCUMENT_NAME,
        DocumentType.TEXT,
        document_id,
        sections,
    )


def render_reconciliation_evidence(line: ReconciliationLine) -> str:
    """Present a single locally calculated line without internal identifiers."""
    values = [f"Reconciliation status: {line.status.value.title()}"]
    if line.po_number:
        values.append(f"PO: {line.po_number}")
    if line.invoice_number:
        values.append(f"Invoice: {line.invoice_number}")
    if line.item_name:
        values.append(f"Item: {line.item_name}")
    if line.invoice:
        if line.invoice_source_line_total is None:
            values.append("Invoice source line total: unavailable")
        else:
            values.append(
                "Invoice source line total: "
                f"${_decimal(line.invoice_source_line_total)}"
            )
    if line.quantity_variance:
        q = line.quantity_variance
        values += [
            f"PO quantity: {_decimal(q.po_quantity)}",
            f"Invoice quantity: {_decimal(q.invoice_quantity)}",
            f"Quantity variance: {_signed(q.variance)}",
        ]
    if line.unit_price_variance:
        p = line.unit_price_variance
        values += [
            f"PO unit price: ${_decimal(p.po_amount)}",
            f"Invoice unit price: ${_decimal(p.invoice_amount)}",
            f"Unit price variance: ${_signed(p.variance)}",
        ]
    if line.invoice_line_total_variance:
        total = line.invoice_line_total_variance
        values += [
            "Calculated invoice extended total: "
            f"${_decimal(total.po_amount)}",
            "Invoice line-total consistency variance: "
            f"${_signed(total.variance)}",
        ]
    if line.extended_variance:
        values.append(f"Extended variance: ${_signed(line.extended_variance.variance)}")
    if line.issue_codes:
        values.append("Issues: " + ", ".join(code.value for code in line.issue_codes))
    return "\n".join(values)


def _matching_candidates(invoice: StructuredRecord, candidates: list[StructuredRecord]):
    vendor = _value(invoice, "vendor") or _value(invoice, "vendor_name")
    if vendor:
        candidates = [
            p
            for p in candidates
            if (po_vendor := _value(p, "vendor") or _value(p, "vendor_name"))
            and _key(po_vendor) == _key(vendor)
        ]
    sku = _value(invoice, "sku")
    if sku:
        selected = [p for p in candidates if _key(_value(p, "sku") or "") == _key(sku)]
        return selected, ReconciliationIssueCode.AMBIGUOUS_MATCH if len(
            selected
        ) > 1 else None
    item = _value(invoice, "item") or _value(invoice, "product_name")
    if not item:
        return [], ReconciliationIssueCode.UNSUPPORTED_SCHEMA
    selected = [
        p
        for p in candidates
        if _key(_value(p, "item") or _value(p, "product_name") or "") == _key(item)
    ]
    return selected, ReconciliationIssueCode.AMBIGUOUS_MATCH if len(
        selected
    ) > 1 else None


def _compare(
    invoice: StructuredRecord, po: StructuredRecord, tolerance: Decimal
) -> ReconciliationLine:
    issues: list[ReconciliationIssueCode] = []
    unit_invoice, unit_po = _value(invoice, "unit"), _value(po, "unit")
    units_conflict = bool(
        unit_invoice and unit_po and _key(unit_invoice) != _key(unit_po)
    )
    if units_conflict:
        issues.append(ReconciliationIssueCode.UNIT_MISMATCH)
    iq, iq_issue = _decimal_field(invoice, "quantity")
    pq, pq_issue = _decimal_field(po, "quantity")
    ip, ip_issue = _decimal_field(invoice, "unit_price")
    pp, pp_issue = _decimal_field(po, "unit_price")
    total, total_issue = _decimal_field(invoice, "line_total")
    issues.extend(
        code for code in (iq_issue, pq_issue, ip_issue, pp_issue, total_issue) if code
    )
    qv = None if units_conflict or iq is None or pq is None else _quantity(iq, pq)
    pv = None if units_conflict or ip is None or pp is None else _money(ip, pp)
    invoice_total_check = (
        None if total is None or iq is None or ip is None else _money(total, iq * ip)
    )
    extended = (
        None
        if units_conflict or iq is None or pq is None or ip is None or pp is None
        else _money(iq * ip, pq * pp)
    )
    invalid_required_input = any(
        code is not None
        for code in (iq_issue, pq_issue, ip_issue, pp_issue, total_issue)
    )
    has_variance = (
        (qv is not None and qv.variance != 0)
        or (pv is not None and abs(pv.variance) > tolerance)
        or (extended is not None and abs(extended.variance) > tolerance)
        or (
            invoice_total_check is not None
            and abs(invoice_total_check.variance) > tolerance
        )
    )
    status = (
        ReconciliationStatus.UNMATCHED
        if units_conflict or invalid_required_input
        else ReconciliationStatus.VARIANCE
        if has_variance
        else ReconciliationStatus.MATCHED
    )
    return _line(
        invoice,
        po,
        status,
        tuple(dict.fromkeys(issues)),
        qv,
        pv,
        invoice_total_check,
        extended,
    )


def _line(
    invoice,
    po,
    status,
    issues,
    qv=None,
    pv=None,
    invoice_total=None,
    extended=None,
    ambiguity_candidate_po_record_ids: tuple[str, ...] = (),
):
    invoice_number = _value(invoice, "invoice_number") if invoice else None
    po_number = (
        _value(invoice, "po_number")
        if invoice
        else _value(po, "po_number")
        if po
        else None
    )
    item = _value(invoice, "item") if invoice else _value(po, "item") if po else None
    item = item or (
        _value(invoice, "product_name")
        if invoice
        else _value(po, "product_name")
        if po
        else None
    )
    sku = _value(invoice, "sku") if invoice else _value(po, "sku") if po else None
    key = "|".join(
        (
            invoice.record_id if invoice else "",
            po.record_id if po else "",
            status.value,
            ",".join(code.value for code in issues),
        )
    )
    return ReconciliationLine(
        hashlib.sha256(key.encode()).hexdigest(),
        status,
        issues,
        _ref(invoice),
        _ref(po),
        invoice_number,
        po_number,
        item,
        sku,
        _value(invoice, "unit") if invoice else _value(po, "unit") if po else None,
        _value(invoice, "unit"),
        _value(po, "unit"),
        _decimal_field(invoice, "line_total")[0] if invoice else None,
        qv,
        pv,
        invoice_total,
        extended,
        ambiguity_candidate_po_record_ids,
    )


def _ref(record: StructuredRecord | None) -> MatchedRecordReference | None:
    if not record:
        return None
    return MatchedRecordReference(
        record.document_id,
        record.document_name,
        record.record_id,
        record.source_locator.sheet_name,
        record.source_locator.row_number,
        record.source_locator.record_label,
    )


def _value(record: StructuredRecord | None, field: str) -> str | None:
    if not record:
        return None
    return next(
        (f.original_value for f in record.fields if f.canonical_name == field), None
    )


def _decimal_field(record, field):
    raw = _value(record, field)
    if raw is None:
        return None, None
    try:
        return Decimal(raw.replace(",", "").replace("$", "").strip()), None
    except InvalidOperation:
        return (
            None,
            ReconciliationIssueCode.INVALID_LINE_TOTAL
            if field == "line_total"
            else ReconciliationIssueCode.INVALID_QUANTITY
            if field == "quantity"
            else ReconciliationIssueCode.INVALID_UNIT_PRICE,
        )


def _quantity(invoice: Decimal, po: Decimal) -> QuantityVariance:
    variance = invoice - po
    return QuantityVariance(
        invoice,
        po,
        variance,
        "equal" if not variance else "over" if variance > 0 else "under",
    )


def _money(invoice: Decimal, po: Decimal) -> MoneyVariance:
    variance = invoice - po
    return MoneyVariance(
        invoice,
        po,
        variance,
        "equal" if not variance else "over" if variance > 0 else "under",
    )


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _decimal(value: Decimal) -> str:
    return f"{value:.2f}" if value == value.quantize(Decimal("0.01")) else str(value)


def _signed(value: Decimal) -> str:
    return ("+" if value >= 0 else "-") + _decimal(abs(value))


def _label(line: ReconciliationLine) -> str:
    return f"{line.invoice_number or 'PO line'} / {line.po_number or 'unmatched'}"
