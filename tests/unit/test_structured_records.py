from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from openpyxl import Workbook

from ai_knowledge_assistant.models import DocumentType, StructuredRecordType
from ai_knowledge_assistant.structured_records import (
    MAX_COLUMNS,
    StructuredRecordError,
    parse_structured_document,
    render_evidence,
    structured_evidence,
)
from ai_knowledge_assistant.uploads import UploadValidationError, accept_upload
from ai_knowledge_assistant.workspace import UploadWorkspace


def _accepted(workspace: UploadWorkspace, name: str, content: bytes):
    return accept_upload(workspace, name, content)


def _xlsx(rows: list[list[object]], extra_sheet: bool = False) -> bytes:
    workbook = Workbook()
    workbook.active.title = "Records"
    for row in rows:
        workbook.active.append(row)
    if extra_sheet:
        workbook.create_sheet("Empty")
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_csv_invoice_bom_normalizes_aliases_and_preserves_source_values(
    tmp_path, caplog
) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    accepted = _accepted(
        workspace,
        "invoices.csv",
        (
            "\ufeffInvoice #,Vendor,Item,Unit Price\n"
            "INV-1048,Granite,London Dry Gin,$24.50\n\n"
        ).encode(),
    )
    document = parse_structured_document(
        accepted, (workspace.uploads_dir / accepted.stored_filename).read_bytes()
    )
    record = document.sheets[0].records[0]
    assert record.record_type is StructuredRecordType.INVOICE
    assert record.fields[0].canonical_name == "invoice_number"
    assert record.fields[3].original_value == "$24.50"
    assert record.fields[3].normalized_value == "24.50"
    assert "Invoice Number: INV-1048" in render_evidence(record)
    assert structured_evidence(document).sections[0].source_locator.row_number == 2
    assert not caplog.records


def test_csv_duplicate_headers_and_binary_are_rejected(tmp_path) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    for name, content in (("bad.csv", b"Item,Item\na,b\n"), ("binary.csv", b"a\x00b")):
        accepted = (
            _accepted(workspace, name, content) if b"\x00" not in content else None
        )
        if accepted:
            with pytest.raises(StructuredRecordError):
                parse_structured_document(accepted, content)
        else:
            with pytest.raises(Exception):
                _accepted(workspace, name, content)
    with pytest.raises(Exception):
        _accepted(workspace, "control.csv", b"column\nvalue\x01\n")


def test_xlsx_multiple_sheets_empty_sheets_and_formula_cells(tmp_path) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    content = _xlsx(
        [
            ["PO Number", "Vendor", "Item", "Quantity"],
            ["PO-221", "Harbor Provisions", "Tonic", 24],
        ],
        True,
    )
    accepted = _accepted(workspace, "orders.xlsx", content)
    document = parse_structured_document(accepted, content)
    assert len(document.sheets) == 1
    assert (
        document.sheets[0].records[0].record_type is StructuredRecordType.PURCHASE_ORDER
    )
    formula = _xlsx([["SKU", "Product Name", "Price"], ["GIN-001", "Gin", "=1+1"]])
    formula_doc = parse_structured_document(
        _accepted(workspace, "formula.xlsx", formula), formula
    )
    assert formula_doc.sheets[0].records[0].fields[2].original_value is None


def test_xlsx_dimension_limit_and_generic_fallback(tmp_path) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    content = _xlsx(
        [[f"field{i}" for i in range(MAX_COLUMNS + 1)], list(range(MAX_COLUMNS + 1))]
    )
    accepted = _accepted(workspace, "wide.xlsx", content)
    with pytest.raises(StructuredRecordError):
        parse_structured_document(accepted, content)
    generic = b"A,B\n1,2\n"
    record = (
        parse_structured_document(_accepted(workspace, "generic.csv", generic), generic)
        .sheets[0]
        .records[0]
    )
    assert record.record_type is StructuredRecordType.GENERIC_TABULAR


def test_structured_document_types_are_accepted(tmp_path) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    assert _accepted(workspace, "a.csv", b"a\nb\n").document_type is DocumentType.CSV
    assert (
        _accepted(workspace, "a.xlsx", _xlsx([["a"], ["b"]])).document_type
        is DocumentType.XLSX
    )


@pytest.mark.parametrize("member", ["xl/vbaProject.bin", "xl/externalLinks/link1.xml"])
def test_xlsx_macros_and_external_links_are_rejected(tmp_path, member: str) -> None:
    source = _xlsx([["SKU"], ["GIN-001"]])
    output = BytesIO()
    with (
        ZipFile(BytesIO(source)) as input_archive,
        ZipFile(output, "w", ZIP_DEFLATED) as archive,
    ):
        for info in input_archive.infolist():
            archive.writestr(info, input_archive.read(info.filename))
        archive.writestr(member, b"synthetic-only")

    with pytest.raises(UploadValidationError):
        _accepted(UploadWorkspace.create(tmp_path), "unsafe.xlsx", output.getvalue())
