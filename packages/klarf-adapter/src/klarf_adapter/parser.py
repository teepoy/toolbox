from __future__ import annotations

from pathlib import Path
from typing import Iterable

from klarf_adapter.model import (
    KlarfDefectList,
    KlarfDefectRecordSpec,
    KlarfDocument,
    KlarfRecord,
    KlarfStatement,
    KlarfTable,
)


TABLE_ROW_WIDTHS = {
    "ClassLookup": 2,
    "SampleTestPlan": 2,
}


class KlarfParseError(ValueError):
    pass


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    in_quotes = False
    escape = False

    for char in text:
        if in_quotes:
            if escape:
                current.append(char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                tokens.append("".join(current))
                current.clear()
                in_quotes = False
                continue
            current.append(char)
            continue

        if char == '"':
            if current:
                tokens.append("".join(current))
                current.clear()
            in_quotes = True
            continue

        if char == ";":
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(";")
            continue

        if char.isspace():
            if current:
                tokens.append("".join(current))
                current.clear()
            continue

        current.append(char)

    if in_quotes:
        raise KlarfParseError("unterminated quoted string")
    if current:
        tokens.append("".join(current))
    return tokens


def loads(text: str) -> KlarfDocument:
    tokens = tokenize(text)
    records: list[KlarfRecord] = []
    index = 0
    current_defect_record_spec: KlarfDefectRecordSpec | None = None

    while index < len(tokens):
        token = tokens[index]
        if token == ";":
            index += 1
            continue

        if token in TABLE_ROW_WIDTHS:
            table, index = _parse_table(token, tokens, index + 1)
            records.append(table)
            continue

        if token == "DefectRecordSpec":
            spec, index = _parse_defect_record_spec(tokens, index + 1)
            records.append(spec)
            current_defect_record_spec = spec
            continue

        if token == "DefectList":
            defect_list, index = _parse_defect_list(
                tokens,
                index + 1,
                row_width=len(current_defect_record_spec.fields)
                if current_defect_record_spec is not None
                else None,
            )
            records.append(defect_list)
            continue

        statement, index = _parse_statement(token, tokens, index + 1)
        records.append(statement)

    return KlarfDocument(records=records)


def load(path: str | Path) -> KlarfDocument:
    return loads(Path(path).read_text(encoding="utf-8"))


def read_klarf(path: str | Path) -> KlarfDocument:
    return load(path)


def dumps(document: KlarfDocument) -> str:
    lines: list[str] = []
    for record in document.records:
        lines.extend(_dump_record(record))
    return "\n".join(lines) + ("\n" if lines else "")


def write_klarf(document: KlarfDocument, path: str | Path) -> None:
    Path(path).write_text(dumps(document), encoding="utf-8")


def _parse_statement(
    name: str, tokens: list[str], index: int
) -> tuple[KlarfStatement, int]:
    values: list[str] = []
    while index < len(tokens) and tokens[index] != ";":
        values.append(tokens[index])
        index += 1
    if index >= len(tokens):
        raise KlarfParseError(f"missing ';' terminator for statement {name!r}")
    return KlarfStatement(name=name, values=values), index + 1


def _parse_table(name: str, tokens: list[str], index: int) -> tuple[KlarfTable, int]:
    if index >= len(tokens):
        raise KlarfParseError(f"missing row count for table {name!r}")

    try:
        row_count = int(tokens[index])
    except ValueError as exc:
        raise KlarfParseError(
            f"invalid row count for table {name!r}: {tokens[index]!r}"
        ) from exc

    index += 1
    row_width = TABLE_ROW_WIDTHS[name]
    rows: list[list[str]] = []

    for row_number in range(row_count):
        if index + row_width > len(tokens):
            raise KlarfParseError(f"table {name!r} ended before row {row_number + 1}")
        row = tokens[index : index + row_width]
        if ";" in row:
            raise KlarfParseError(f"table {name!r} ended before row {row_number + 1}")
        rows.append(row)
        index += row_width

    if index < len(tokens) and tokens[index] == ";":
        index += 1

    return KlarfTable(name=name, row_width=row_width, rows=rows), index


def _parse_defect_record_spec(
    tokens: list[str], index: int
) -> tuple[KlarfDefectRecordSpec, int]:
    if index >= len(tokens):
        raise KlarfParseError("missing field count for DefectRecordSpec")

    try:
        field_count = int(tokens[index])
    except ValueError as exc:
        raise KlarfParseError(
            f"invalid field count for DefectRecordSpec: {tokens[index]!r}"
        ) from exc

    index += 1
    fields = tokens[index : index + field_count]
    if len(fields) != field_count or ";" in fields:
        raise KlarfParseError("DefectRecordSpec ended before all fields were read")

    index += field_count
    if index >= len(tokens) or tokens[index] != ";":
        raise KlarfParseError("missing ';' terminator for DefectRecordSpec")

    return KlarfDefectRecordSpec(fields=fields), index + 1


def _parse_defect_list(
    tokens: list[str], index: int, row_width: int | None
) -> tuple[KlarfDefectList, int]:
    values: list[str] = []
    while index < len(tokens) and tokens[index] != ";":
        values.append(tokens[index])
        index += 1
    if index >= len(tokens):
        raise KlarfParseError("missing ';' terminator for DefectList")

    if row_width is None or row_width <= 0:
        rows = [values] if values else []
        return KlarfDefectList(rows=rows), index + 1

    if len(values) % row_width != 0:
        raise KlarfParseError(
            f"DefectList contains {len(values)} values which is not divisible by row width {row_width}"
        )

    rows = [
        values[offset : offset + row_width]
        for offset in range(0, len(values), row_width)
    ]
    return KlarfDefectList(rows=rows), index + 1


def _dump_record(record: KlarfRecord) -> Iterable[str]:
    if isinstance(record, KlarfStatement):
        yield _render_statement(record.name, record.values)
        return

    if isinstance(record, KlarfTable):
        yield f"{record.name} {len(record.rows)}"
        for row in record.rows:
            if len(row) != record.row_width:
                raise ValueError(
                    f"row {row!r} does not match width {record.row_width} for {record.name}"
                )
            yield _render_values(row)
        yield ";"
        return

    if isinstance(record, KlarfDefectRecordSpec):
        yield _render_statement(
            "DefectRecordSpec", [str(len(record.fields)), *record.fields]
        )
        return

    if isinstance(record, KlarfDefectList):
        if len(record.rows) <= 1:
            yield _render_statement("DefectList", record.values)
            return

        yield "DefectList"
        for row in record.rows:
            yield f" {_render_values(row)}"
        yield ";"
        return

    raise TypeError(f"unsupported KLARF record type: {type(record)!r}")


def _render_statement(name: str, values: list[str]) -> str:
    if values:
        return f"{name} {_render_values(values)};"
    return f"{name};"


def _render_values(values: list[str]) -> str:
    return " ".join(_quote(value) for value in values)


def _quote(value: str) -> str:
    if not value:
        return '""'
    if any(char.isspace() for char in value) or ";" in value or '"' in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value
