"""Offline tests for deterministic invoice/PO reconciliation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ai_knowledge_assistant.models import ReconciliationIssueCode, ReconciliationStatus
from ai_knowledge_assistant.reconciliation import (
    reconcile,
    reconciliation_evidence,
    render_reconciliation_evidence,
)
from ai_knowledge_assistant.structured_records import parse_structured_document
from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.workspace import UploadWorkspace


def _records(tmp_path, invoices: str, pos: str):
    workspace = UploadWorkspace.create(tmp_path)
    invoice = accept_upload(workspace, "invoices.csv", invoices.encode())
    po = accept_upload(workspace, "purchase_orders.csv", pos.encode())
    return tuple(
        record
        for document, content in ((invoice, invoices), (po, pos))
        for sheet in parse_structured_document(document, content.encode()).sheets
        for record in sheet.records
    )


HEADER_I = (
    "Invoice Number,PO Number,Vendor,SKU,Item,Quantity,Unit Price,Line Total,Unit\n"
)
HEADER_P = "PO Number,Vendor,SKU,Item,Quantity,Unit Price,Unit\n"


def test_exact_sku_match_uses_decimal_variance_and_path_free_provenance(
    tmp_path,
) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-1,PO-1,Vendor,A,Item,6,24.50,147.00,bottle\n",
            HEADER_P + "PO-1,Vendor,A,Item,6,24.00,bottle\n",
        )
    )
    line = result.lines[0]

    assert line.status is ReconciliationStatus.VARIANCE
    assert line.quantity_variance.variance == Decimal("0")  # type: ignore[union-attr]
    assert line.unit_price_variance.variance == Decimal("0.50")  # type: ignore[union-attr]
    assert line.extended_variance.variance == Decimal("3.00")  # type: ignore[union-attr]
    assert line.invoice and line.invoice.row_number == 2
    assert line.purchase_order and line.purchase_order.row_number == 2
    assert "/" not in line.invoice.document_name  # type: ignore[union-attr]
    assert "Extended variance: $+3.00" in render_reconciliation_evidence(line)


def test_tolerance_does_not_round_or_hide_the_underlying_amount(tmp_path) -> None:
    records = _records(
        tmp_path,
        HEADER_I + "INV-1,PO-1,V,A,Item,1,10.01,10.01,each\n",
        HEADER_P + "PO-1,V,A,Item,1,10.00,each\n",
    )
    line = reconcile(records).lines[0]

    assert line.status is ReconciliationStatus.MATCHED
    assert line.unit_price_variance.variance == Decimal("0.01")  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("invoice_price", "expected_status"),
    (
        (Decimal("10.00"), ReconciliationStatus.MATCHED),
        (Decimal("10.01"), ReconciliationStatus.MATCHED),
        (Decimal("9.99"), ReconciliationStatus.MATCHED),
        (Decimal("10.0101"), ReconciliationStatus.VARIANCE),
        (Decimal("9.9899"), ReconciliationStatus.VARIANCE),
        (Decimal("12.00"), ReconciliationStatus.VARIANCE),
        (Decimal("8.00"), ReconciliationStatus.VARIANCE),
    ),
)
def test_monetary_tolerance_is_inclusive_at_one_cent_and_retains_exact_decimals(
    tmp_path, invoice_price: Decimal, expected_status: ReconciliationStatus
) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I
            + f"INV-1,PO-1,V,A,Item,1,{invoice_price},{invoice_price},each\n",
            HEADER_P + "PO-1,V,A,Item,1,10.00,each\n",
        )
    )
    line = next(line for line in result.lines if line.invoice is not None)

    assert line.status is expected_status
    assert line.unit_price_variance and line.unit_price_variance.variance == (
        invoice_price - Decimal("10.00")
    )


def test_ambiguous_and_unit_mismatch_are_never_guessed(tmp_path) -> None:
    ambiguous_records = _records(
        tmp_path,
        HEADER_I + "INV-1,PO-1,V,,Item,1,10,10,each\n",
        HEADER_P + "PO-1,V,A,Item,1,10,each\nPO-1,V,B,Item,1,10,each\n",
    )
    ambiguous = reconcile(ambiguous_records)
    mismatch = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-2,PO-2,V,A,Item,5,2,10,case\n",
            HEADER_P + "PO-2,V,A,Item,5,2,bottle\n",
        )
    )

    ambiguous_line = next(
        line
        for line in ambiguous.lines
        if line.status is ReconciliationStatus.UNMATCHED
    )
    mismatch_line = next(line for line in mismatch.lines if line.invoice is not None)
    assert ReconciliationIssueCode.AMBIGUOUS_MATCH in ambiguous_line.issue_codes
    assert ambiguous_line.ambiguity_candidate_po_record_ids == tuple(
        record.record_id
        for record in ambiguous_records
        if record.record_type.value == "purchase_order"
    )
    assert mismatch_line.status is ReconciliationStatus.UNMATCHED
    assert ReconciliationIssueCode.UNIT_MISMATCH in mismatch_line.issue_codes
    assert mismatch_line.quantity_variance is None
    assert mismatch_line.unit_price_variance is None
    assert mismatch_line.extended_variance is None


def test_ambiguity_provenance_preserves_only_sku_and_vendor_filtered_candidates(
    tmp_path,
) -> None:
    records = _records(
        tmp_path,
        HEADER_I + "INV-1,PO-1,Vendor A,SKU-1,Item,1,10,10,each\n",
        HEADER_P
        + "PO-1,Vendor A,SKU-1,Item,1,10,each\n"
        + "PO-1,Vendor A,SKU-1,Item,1,10,each\n"
        + "PO-1,Vendor B,SKU-1,Item,1,10,each\n",
    )
    result = reconcile(records)
    ambiguous = next(line for line in result.lines if line.invoice is not None)
    po_ids = [
        record.record_id
        for record in records
        if record.record_type.value == "purchase_order"
    ]

    assert ambiguous.ambiguity_candidate_po_record_ids == tuple(po_ids[:2])
    assert po_ids[2] not in ambiguous.ambiguity_candidate_po_record_ids
    assert result.summary.missing_on_invoice_count == 3


def test_missing_lines_malformed_values_and_evidence_are_safe(tmp_path) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-1,PO-1,V,A,Extra,2,bad,4,each\n",
            HEADER_P + "PO-1,V,B,Absent,2,2,each\n",
        )
    )
    statuses = {line.status for line in result.lines}
    evidence = reconciliation_evidence(result)

    assert ReconciliationStatus.MISSING_ON_PO in statuses
    assert ReconciliationStatus.MISSING_ON_INVOICE in statuses
    assert any(
        ReconciliationIssueCode.ITEM_NOT_FOUND in line.issue_codes
        for line in result.lines
    )
    assert all("/" not in section.text for section in evidence.sections)
    assert all(len(section.section_id) == 64 for section in evidence.sections)


def test_consumed_po_line_cannot_be_reused_by_another_invoice_line(tmp_path) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I
            + "INV-1,PO-1,V,A,Item,1,10,10,each\n"
            + "INV-2,PO-1,V,A,Item,1,10,10,each\n",
            HEADER_P + "PO-1,V,A,Item,1,10,each\n",
        )
    )
    invoice_lines = [line for line in result.lines if line.invoice is not None]

    assert [line.invoice_number for line in invoice_lines if line.purchase_order] == [
        "INV-1"
    ]
    assert any(
        line.invoice_number == "INV-2"
        and line.status is ReconciliationStatus.MISSING_ON_PO
        and ReconciliationIssueCode.ITEM_NOT_FOUND in line.issue_codes
        for line in invoice_lines
    )
    assert result.summary.monetary_variance_line_count == 1


@pytest.mark.parametrize(
    ("po_vendor", "expected_status"),
    (
        ("", ReconciliationStatus.MISSING_ON_PO),
        ("Vendor", ReconciliationStatus.MATCHED),
        ("Other Vendor", ReconciliationStatus.MISSING_ON_PO),
    ),
)
def test_vendor_context_requires_exact_non_blank_po_vendor(
    tmp_path, po_vendor: str, expected_status: ReconciliationStatus
) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-1,PO-1,Vendor,A,Item,1,10,10,each\n",
            HEADER_P + f"PO-1,{po_vendor},A,Item,1,10,each\n",
        )
    )
    line = next(line for line in result.lines if line.invoice is not None)

    assert line.status is expected_status
    assert (line.purchase_order is not None) is (
        expected_status is ReconciliationStatus.MATCHED
    )


@pytest.mark.parametrize(
    ("invoice_quantity", "invoice_price", "expected_issue"),
    (
        ("bad", "10", ReconciliationIssueCode.INVALID_QUANTITY),
        ("1", "bad", ReconciliationIssueCode.INVALID_UNIT_PRICE),
    ),
)
def test_invalid_required_quantity_or_price_is_unresolved_and_not_aggregated(
    tmp_path, invoice_quantity: str, invoice_price: str, expected_issue
) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I
            + f"INV-1,PO-1,V,A,Item,{invoice_quantity},{invoice_price},10,each\n",
            HEADER_P + "PO-1,V,A,Item,1,10,each\n",
        )
    )
    line = next(line for line in result.lines if line.invoice is not None)

    assert line.status is ReconciliationStatus.UNMATCHED
    assert expected_issue in line.issue_codes
    assert line.extended_variance is None
    assert result.summary.total_monetary_variance == Decimal("0")


def test_source_and_calculated_line_totals_are_distinct_in_evidence(tmp_path) -> None:
    result = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-1,PO-1,V,A,Item,2,3,6,each\n",
            HEADER_P + "PO-1,V,A,Item,2,3,each\n",
        )
    )
    line = next(line for line in result.lines if line.invoice is not None)
    evidence = render_reconciliation_evidence(line)

    assert line.invoice_source_line_total == Decimal("6")
    assert line.invoice_line_total_variance
    assert "Invoice source line total: $6.00" in evidence
    assert "Calculated invoice extended total: $6.00" in evidence
    assert "Invoice line-total consistency variance: $+0.00" in evidence


def test_absent_or_malformed_source_line_total_is_not_used_for_consistency_check(
    tmp_path,
) -> None:
    absent = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-1,PO-1,V,A,Item,2,3,,each\n",
            HEADER_P + "PO-1,V,A,Item,2,3,each\n",
        )
    )
    malformed = reconcile(
        _records(
            tmp_path,
            HEADER_I + "INV-2,PO-2,V,B,Item,2,3,nope,each\n",
            HEADER_P + "PO-2,V,B,Item,2,3,each\n",
        )
    )
    absent_line = next(line for line in absent.lines if line.invoice is not None)
    malformed_line = next(line for line in malformed.lines if line.invoice is not None)

    assert absent_line.invoice_source_line_total is None
    assert absent_line.invoice_line_total_variance is None
    assert absent_line.extended_variance and absent_line.extended_variance.variance == 0
    assert "Invoice source line total: unavailable" in render_reconciliation_evidence(
        absent_line
    )
    assert malformed_line.invoice_line_total_variance is None
    assert ReconciliationIssueCode.INVALID_LINE_TOTAL in malformed_line.issue_codes
    assert malformed_line.status is ReconciliationStatus.UNMATCHED
    assert malformed.summary.total_monetary_variance == Decimal("0")
