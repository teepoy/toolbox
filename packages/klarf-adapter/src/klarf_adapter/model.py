from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Scalar = str


@dataclass(slots=True)
class KlarfStatement:
    name: str
    values: list[Scalar] = field(default_factory=list)

    kind: Literal["statement"] = field(init=False, default="statement")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KlarfStatement":
        return cls(
            name=str(data["name"]),
            values=[str(value) for value in data.get("values", [])],
        )


@dataclass(slots=True)
class KlarfTable:
    name: str
    row_width: int
    rows: list[list[Scalar]] = field(default_factory=list)

    kind: Literal["table"] = field(init=False, default="table")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "row_width": self.row_width,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KlarfTable":
        rows = [[str(value) for value in row] for row in data.get("rows", [])]
        return cls(name=str(data["name"]), row_width=int(data["row_width"]), rows=rows)


@dataclass(slots=True)
class KlarfDefectRecordSpec:
    fields: list[str]

    kind: Literal["defect_record_spec"] = field(
        init=False, default="defect_record_spec"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fields": self.fields,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KlarfDefectRecordSpec":
        return cls(fields=[str(value) for value in data.get("fields", [])])


@dataclass(slots=True)
class KlarfDefectList:
    rows: list[list[Scalar]] = field(default_factory=list)

    kind: Literal["defect_list"] = field(init=False, default="defect_list")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "row_width": self.row_width,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KlarfDefectList":
        if "rows" in data:
            rows = [[str(value) for value in row] for row in data.get("rows", [])]
            return cls(rows=rows)

        values = [str(value) for value in data.get("values", [])]
        return cls(rows=[values] if values else [])

    @property
    def row_width(self) -> int:
        if not self.rows:
            return 0
        return len(self.rows[0])

    @property
    def values(self) -> list[Scalar]:
        if not self.rows:
            return []
        if len(self.rows) == 1:
            return self.rows[0]
        return [value for row in self.rows for value in row]

    def as_mapping(
        self, field_names: list[str], row_index: int = 0
    ) -> dict[str, str | list[str]]:
        return self.as_mappings(field_names)[row_index]

    def as_mappings(self, field_names: list[str]) -> list[dict[str, str | list[str]]]:
        if not field_names:
            raise ValueError("field_names must not be empty")

        mappings: list[dict[str, str | list[str]]] = []
        for row in self.rows:
            mapping: dict[str, str | list[str]] = {}
            for index, field_name in enumerate(field_names):
                if index >= len(row):
                    mapping[field_name] = ""
                    continue

                if index == len(field_names) - 1:
                    tail = row[index:]
                    mapping[field_name] = tail[0] if len(tail) == 1 else tail
                    break

                mapping[field_name] = row[index]
            mappings.append(mapping)

        return mappings


KlarfRecord = KlarfStatement | KlarfTable | KlarfDefectRecordSpec | KlarfDefectList


@dataclass(slots=True)
class KlarfDocument:
    records: list[KlarfRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KlarfDocument":
        records: list[KlarfRecord] = []
        for raw_record in data.get("records", []):
            kind = raw_record.get("kind")
            if kind == "statement":
                records.append(KlarfStatement.from_dict(raw_record))
            elif kind == "table":
                records.append(KlarfTable.from_dict(raw_record))
            elif kind == "defect_record_spec":
                records.append(KlarfDefectRecordSpec.from_dict(raw_record))
            elif kind == "defect_list":
                records.append(KlarfDefectList.from_dict(raw_record))
            else:
                raise ValueError(f"unsupported KLARF record kind: {kind!r}")
        return cls(records=records)

    def statements_named(self, name: str) -> list[KlarfStatement]:
        return [
            record
            for record in self.records
            if isinstance(record, KlarfStatement) and record.name == name
        ]
