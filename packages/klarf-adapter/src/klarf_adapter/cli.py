from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from klarf_adapter.model import KlarfDocument
from klarf_adapter.parser import dumps, load, write_klarf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and generate KLARF files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser(
        "read", help="Parse a KLARF file into YAML or JSON"
    )
    read_parser.add_argument("input", type=Path)
    read_parser.add_argument("-o", "--output", type=Path)
    read_parser.add_argument("--format", choices=("yaml", "json"), default="yaml")

    write_parser = subparsers.add_parser(
        "write", help="Generate a KLARF file from YAML or JSON"
    )
    write_parser.add_argument("input", type=Path)
    write_parser.add_argument("-o", "--output", type=Path)
    write_parser.add_argument("--input-format", choices=("yaml", "json"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "read":
        return _read_command(args.input, args.output, args.format)

    if args.command == "write":
        return _write_command(args.input, args.output, args.input_format)

    parser.error(f"unknown command: {args.command}")
    return 2


def _read_command(
    input_path: Path, output_path: Path | None, output_format: str
) -> int:
    document = load(input_path)
    payload = document.to_dict()

    if output_format == "json":
        rendered = json.dumps(payload, indent=2) + "\n"
    else:
        rendered = yaml.safe_dump(payload, sort_keys=False)

    if output_path is None:
        sys.stdout.write(rendered)
    else:
        output_path.write_text(rendered, encoding="utf-8")
    return 0


def _write_command(
    input_path: Path, output_path: Path | None, input_format: str | None
) -> int:
    payload = _read_payload(input_path, input_format)
    document = KlarfDocument.from_dict(payload)
    rendered = dumps(document)

    if output_path is None:
        sys.stdout.write(rendered)
    else:
        write_klarf(document, output_path)
    return 0


def _read_payload(path: Path, input_format: str | None) -> dict[str, Any]:
    selected_format = input_format or _infer_format(path)
    content = path.read_text(encoding="utf-8")

    if selected_format == "json":
        return json.loads(content)
    if selected_format == "yaml":
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            raise ValueError("YAML input must decode to a mapping")
        return payload
    raise ValueError(f"unsupported input format: {selected_format!r}")


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    return "yaml"


if __name__ == "__main__":
    raise SystemExit(main())
