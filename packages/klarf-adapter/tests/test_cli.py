from __future__ import annotations

import json
from pathlib import Path

import yaml

from klarf_adapter.cli import main


SAMPLE_KLARF = """LotID LOT-42;
ClassLookup 1
17 \"Top Side Chipping\"
;
DefectRecordSpec 2 DEFECTID CLASSNUMBER;
DefectList 327 17;
"""


def test_cli_read_writes_yaml(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "sample.klarf"
    input_path.write_text(SAMPLE_KLARF, encoding="utf-8")

    exit_code = main(["read", str(input_path)])
    assert exit_code == 0

    output = capsys.readouterr().out
    payload = yaml.safe_load(output)
    assert payload["records"][0]["name"] == "LotID"


def test_cli_write_generates_klarf(tmp_path: Path) -> None:
    payload = {
        "records": [
            {"kind": "statement", "name": "LotID", "values": ["LOT-42"]},
            {
                "kind": "table",
                "name": "ClassLookup",
                "row_width": 2,
                "rows": [["17", "Top Side Chipping"]],
            },
            {"kind": "defect_record_spec", "fields": ["DEFECTID", "CLASSNUMBER"]},
            {"kind": "defect_list", "rows": [["327", "17"]]},
        ]
    }
    input_path = tmp_path / "sample.yaml"
    output_path = tmp_path / "sample.klarf"
    input_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exit_code = main(["write", str(input_path), "-o", str(output_path)])
    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == (
        "LotID LOT-42;\n"
        "ClassLookup 1\n"
        '17 "Top Side Chipping"\n'
        ";\n"
        "DefectRecordSpec 2 DEFECTID CLASSNUMBER;\n"
        "DefectList 327 17;\n"
    )


def test_cli_read_json_output(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "sample.klarf"
    input_path.write_text(SAMPLE_KLARF, encoding="utf-8")

    exit_code = main(["read", str(input_path), "--format", "json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"][1]["name"] == "ClassLookup"
