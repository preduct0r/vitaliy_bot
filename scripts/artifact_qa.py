#!/usr/bin/env python3
"""Read-only mechanical QA for recipient-facing office artifacts.

This tool deliberately does not decide whether the contents are correct.  A
``passing`` result is only the mechanical part of R-OUT-007; the separate
semantic review, including contradiction search, remains mandatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".docx": "docx",
    ".docm": "docx",
    ".pdf": "pdf",
    ".pptm": "pptx",
    ".pptx": "pptx",
    ".xlsm": "xlsx",
    ".xlsx": "xlsx",
}
OOXML_REQUIRED_PARTS = {
    "docx": "word/document.xml",
    "pptx": "ppt/presentation.xml",
    "xlsx": "xl/workbook.xml",
}
OOXML_FEATURE_PATHS = {
    "activex": ("activex/",),
    "external_links": ("externalLinks/", "externalLink"),
    "ole": ("embeddings/", "oleObject", "oleobjects/"),
    "pivot_tables": ("pivotTables/", "pivotCache"),
    "signatures": ("_xmlsignatures/", "origin.sigs", "signature"),
    "vba": ("vbaProject.bin",),
}
OOXML_XML_FEATURES = {
    "fields": ("fldchar", "instrtext", "simplefield"),
    "tracked_changes": ("trackrevisions", "<w:ins", "<w:del", "<w:movefrom", "<w:moveto"),
}
PDF_FEATURE_PATTERNS = {
    "forms": rb"/AcroForm\b",
    "signatures": rb"/ByteRange\b|/Type\s*/Sig\b",
}
BLOCKING_FEATURES = {
    "activex",
    "external_links",
    "fields",
    "ole",
    "pivot_tables",
    "signatures",
    "tracked_changes",
    "vba",
}


class ArtifactQaError(RuntimeError):
    """A file cannot be checked safely or structurally."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _tool_version(command: Iterable[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command), check=False, text=True, capture_output=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else None


def _format_for(path: Path) -> str:
    artifact_format = FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if artifact_format is None:
        raise ArtifactQaError(f"unsupported format: {path.suffix or '(no extension)'}")
    return artifact_format


def _ooxml_features(archive: zipfile.ZipFile) -> set[str]:
    names = archive.namelist()
    lowered = {name.lower() for name in names}
    features = {
        feature
        for feature, fragments in OOXML_FEATURE_PATHS.items()
        if any(fragment.lower() in name for fragment in fragments for name in lowered)
    }
    xml_text = ""
    for name in names:
        if name.endswith(".xml") and archive.getinfo(name).file_size <= 5_000_000:
            try:
                xml_text += archive.read(name).decode("utf-8", errors="ignore").lower()
            except KeyError:
                continue
    features.update(
        feature
        for feature, fragments in OOXML_XML_FEATURES.items()
        if any(fragment in xml_text for fragment in fragments)
    )
    return features


def inspect_file(path: Path) -> dict[str, Any]:
    """Inspect structure and editing risks without changing the artifact."""
    path = path.resolve()
    checks: list[dict[str, str]] = []
    warnings: list[str] = []
    blockers: list[str] = []
    features: set[str] = set()
    if not path.is_file():
        return {
            "source": str(path),
            "format": None,
            "checks": [_check("source_exists", "failed", "file does not exist")],
            "features": [],
            "warnings": [],
            "blockers": [],
            "status": "failed",
        }

    try:
        artifact_format = _format_for(path)
    except ArtifactQaError as error:
        return {
            "source": str(path),
            "format": None,
            "sha256": _sha256(path),
            "checks": [_check("format", "failed", str(error))],
            "features": [],
            "warnings": [],
            "blockers": [],
            "status": "failed",
        }

    result: dict[str, Any] = {
        "source": str(path),
        "format": artifact_format,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "checks": checks,
        "features": [],
        "warnings": warnings,
        "blockers": blockers,
        "status": "passing",
    }
    try:
        if artifact_format in OOXML_REQUIRED_PARTS:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                required_part = OOXML_REQUIRED_PARTS[artifact_format]
                if bad_member or required_part not in archive.namelist():
                    detail = (
                        f"corrupt member: {bad_member}"
                        if bad_member
                        else f"missing required part: {required_part}"
                    )
                    checks.append(_check("ooxml_package", "failed", detail))
                    result["status"] = "failed"
                    return result
                checks.append(_check("ooxml_package", "passing", "ZIP package is readable"))
                features = _ooxml_features(archive)
        elif artifact_format == "pdf":
            header = path.read_bytes()[:1024 * 1024]
            if not header.startswith(b"%PDF-"):
                checks.append(_check("pdf_header", "failed", "missing PDF header"))
                result["status"] = "failed"
                return result
            checks.append(_check("pdf_header", "passing", "PDF header detected"))
            features = {
                feature
                for feature, pattern in PDF_FEATURE_PATTERNS.items()
                if re.search(pattern, header)
            }
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                next(csv.reader(source), None)
            checks.append(_check("csv_read", "passing", "UTF-8 CSV is readable"))
            warnings.append("CSV has no native visual layout; visual QA uses a temporary XLSX copy.")
    except (OSError, UnicodeError, zipfile.BadZipFile) as error:
        checks.append(_check("structure", "failed", str(error)))
        result["status"] = "failed"
        return result

    result["features"] = sorted(features)
    blockers.extend(
        f"native feature requires a native Office review before automatic editing: {feature}"
        for feature in sorted(features & BLOCKING_FEATURES)
    )
    if blockers:
        result["status"] = "blocked"
    return result


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False, timeout=180
        )
    except FileNotFoundError as error:
        raise ArtifactQaError(f"required tool is unavailable: {command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise ArtifactQaError(f"tool timed out: {command[0]}") from error


def _csv_to_xlsx(source: Path, target: Path) -> Path:
    try:
        from openpyxl import Workbook
    except ImportError as error:
        raise ArtifactQaError("openpyxl is required to render CSV") from error
    workbook = Workbook()
    sheet = workbook.active
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            sheet.append(row)
    workbook.save(target)
    return target


def _render_pdf(source: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_prefix = output_dir / "page"
    completed = _run(["pdftoppm", "-png", "-r", "144", str(source), str(output_prefix)])
    if completed.returncode:
        raise ArtifactQaError(completed.stderr.strip() or "Poppler could not render the PDF")
    pngs = sorted(output_dir.glob("page-*.png"))
    if not pngs:
        raise ArtifactQaError("Poppler rendered no pages")
    return _image_pages(pngs)


def _image_pages(pngs: list[Path]) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise ArtifactQaError("Pillow is required to verify rendered pages") from error
    pages = []
    for number, image_path in enumerate(pngs, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 1 or height <= 1 or image_path.stat().st_size == 0:
            raise ArtifactQaError(f"empty rendered image: {image_path.name}")
        pages.append(
            {
                "number": number,
                "path": str(image_path),
                "bytes": image_path.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    return pages


def render_file(path: Path, output_dir: Path) -> dict[str, Any]:
    """Render an artifact to PNG pages, using a separate output directory."""
    path = path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_format = _format_for(path)
    with tempfile.TemporaryDirectory(prefix="artifact-qa-") as temporary_name:
        temporary_dir = Path(temporary_name)
        source_for_pdf = path
        if artifact_format == "csv":
            source_for_pdf = _csv_to_xlsx(path, temporary_dir / f"{path.stem}.xlsx")
            artifact_format = "xlsx"
        if artifact_format != "pdf":
            completed = _run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temporary_dir),
                    str(source_for_pdf),
                ]
            )
            pdf_candidate = temporary_dir / f"{source_for_pdf.stem}.pdf"
            if completed.returncode or not pdf_candidate.is_file():
                detail = completed.stderr.strip() or completed.stdout.strip() or "LibreOffice did not create PDF"
                raise ArtifactQaError(detail)
            source_for_pdf = pdf_candidate
        pages = _render_pdf(source_for_pdf, output_dir)
    return {"rendered_from": str(path), "pages": pages, "status": "passing"}


def verify_file(path: Path, output_dir: Path) -> dict[str, Any]:
    """Create the stable artifact_qa.json report and return it."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection = inspect_file(path)
    report: dict[str, Any] = {
        "artifact": inspection,
        "tool_versions": {
            "python": sys.version.split()[0],
            "libreoffice": _tool_version(["soffice", "--version"]),
            "poppler": _tool_version(["pdftoppm", "-v"]),
        },
        "pages": [],
        "warnings": list(inspection["warnings"]),
        "blockers": list(inspection["blockers"]),
        "status": inspection["status"],
    }
    if inspection["status"] != "failed":
        try:
            rendering = render_file(path, output_dir)
            report["pages"] = rendering["pages"]
        except ArtifactQaError as error:
            report["warnings"].append(str(error))
            if report["status"] == "passing":
                report["status"] = "blocked"
    report_path = output_dir / "artifact_qa.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "render", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("file", type=Path)
        if command != "inspect":
            subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_file(args.file)
        elif args.command == "render":
            report = render_file(args.file, args.output_dir)
        else:
            report = verify_file(args.file, args.output_dir)
    except ArtifactQaError as error:
        report = {"status": "failed", "error": str(error)}
    _print_report(report, args.json)
    return 0 if report.get("status") == "passing" else 2


if __name__ == "__main__":
    raise SystemExit(main())
