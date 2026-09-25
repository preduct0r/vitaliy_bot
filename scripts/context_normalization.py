#!/usr/bin/env python3
"""Create local, read-only text views of private weekly-context sources.

The normalizer is deliberately independent of the ledger and server. It runs
only after ``weekly_context`` has validated a claim and writes derived text
only to that run's local temporary directory. Original sources are never
modified, copied to Git, or sent through Mutagen. PDF text extraction may use
an isolated short-lived copy because Poppler accepts a file path.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import posixpath
import re
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


TEXT_SUFFIXES = {
    ".cfg", ".md", ".py", ".rst", ".sh", ".srt", ".toml", ".ts", ".tsx", ".txt", ".vtt", ".yaml", ".yml", ".zsh",
}
HTML_SUFFIXES = {".htm", ".html"}
OFFICE_SUFFIXES = {".docm", ".docx", ".pptm", ".pptx", ".xlsm", ".xlsx"}
VISUAL_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_ARCHIVE_DEPTH = 1
MAX_ARCHIVE_MEMBERS = 500
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_XLSX_ROWS = 20_000
MAX_XLSX_CELLS = 100_000

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


class NormalizationError(RuntimeError):
    """The source cannot safely become a local text view."""


class VisualReviewRequired(NormalizationError):
    """The source has no trustworthy text layer and needs a visual pass."""


class _HtmlText(HTMLParser):
    _BLOCK_TAGS = {"article", "blockquote", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.casefold()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and not item.name.startswith("~$"):
            yield item


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise NormalizationError("text_encoding_unsupported")


def _html_text(payload: bytes) -> str:
    parser = _HtmlText()
    parser.feed(_decode_text(payload))
    parser.close()
    return parser.text()


def _json_text(payload: bytes) -> str:
    try:
        value = json.loads(_decode_text(payload))
    except json.JSONDecodeError as error:
        raise NormalizationError("invalid_json") from error
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _csv_text(payload: bytes) -> str:
    source = io.StringIO(_decode_text(payload))
    sample = source.read(4096)
    source.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    target = io.StringIO()
    writer = csv.writer(target, delimiter="\t", lineterminator="\n")
    # A short sample may contain quoted fields but no escaped quote. Sniffer
    # then guesses doublequote=False and breaks later multiline transcripts.
    for row in csv.reader(source, dialect, doublequote=True):
        writer.writerow(row)
    return target.getvalue()


def _xml_text(payload: bytes, text_tag: str) -> str:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise NormalizationError("invalid_ooxml_xml") from error
    return "".join(element.text or "" for element in root.iter(text_tag))


def _docx_text(archive: zipfile.ZipFile) -> str:
    try:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    except KeyError as error:
        raise NormalizationError("missing_docx_document_xml") from error
    paragraphs = []
    for paragraph in document.iter(WORD_NS + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(WORD_NS + "t"))
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _pptx_text(archive: zipfile.ZipFile) -> str:
    available = set(archive.namelist())
    slides: list[str] = []
    # ``slideN.xml`` is not a presentation order.  The sldIdLst plus the
    # relationship file is the authoritative order for PPTX decks.
    try:
        presentation = ElementTree.fromstring(archive.read("ppt/presentation.xml"))
        relationships = ElementTree.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
        targets = {
            relation.get("Id"): posixpath.normpath(
                relation.get("Target", "").lstrip("/")
                if relation.get("Target", "").startswith("/")
                else posixpath.join("ppt", relation.get("Target", ""))
            )
            for relation in relationships.iter(PACKAGE_REL_NS + "Relationship")
            if relation.get("TargetMode") != "External"
        }
        for slide_id in presentation.iter("{http://schemas.openxmlformats.org/presentationml/2006/main}sldId"):
            target = targets.get(slide_id.get(REL_NS + "id"))
            if target in available and re.fullmatch(r"ppt/slides/slide\d+\.xml", target):
                slides.append(target)
    except (KeyError, ElementTree.ParseError):
        slides = []
    if not slides:
        slides = sorted(
            (name for name in available if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=_natural_key,
        )
    if not slides:
        raise NormalizationError("missing_pptx_slides")
    chunks = []
    for number, name in enumerate(slides, start=1):
        text = _xml_text(archive.read(name), DRAWING_NS + "t")
        chunks.append(f"## Слайд {number}\n{text}")
    return "\n\n".join(chunks)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    except ElementTree.ParseError as error:
        raise NormalizationError("invalid_xlsx_shared_strings") from error
    return ["".join(node.text or "" for node in item.iter(SHEET_NS + "t")) for item in root.iter(SHEET_NS + "si")]


def _xlsx_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError as error:
        raise NormalizationError("missing_xlsx_workbook_xml") from error
    except ElementTree.ParseError as error:
        raise NormalizationError("invalid_xlsx_workbook_xml") from error
    targets = {
        relation.get("Id"): posixpath.normpath(
            relation.get("Target", "").lstrip("/")
            if relation.get("Target", "").startswith("/")
            else posixpath.join("xl", relation.get("Target", ""))
        )
        for relation in relationships.iter(PACKAGE_REL_NS + "Relationship")
        if relation.get("TargetMode") != "External"
    }
    sheets = []
    for sheet in workbook.iter(SHEET_NS + "sheet"):
        target = targets.get(sheet.get(REL_NS + "id"))
        if target and target in archive.namelist():
            sheets.append((sheet.get("name") or target, target))
    if not sheets:
        raise NormalizationError("missing_xlsx_sheets")
    return sheets


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    formula = cell.find(SHEET_NS + "f")
    if formula is not None:
        return "=" + (formula.text or "")
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(SHEET_NS + "t"))
    raw = cell.findtext(SHEET_NS + "v", default="")
    if kind == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return f"[invalid shared string: {raw}]"
    if kind == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def _xlsx_text(archive: zipfile.ZipFile) -> tuple[str, list[str]]:
    shared_strings = _shared_strings(archive)
    chunks: list[str] = []
    warnings: list[str] = []
    total_cells = 0
    for sheet_name, part_name in _xlsx_sheets(archive):
        try:
            root = ElementTree.fromstring(archive.read(part_name))
        except ElementTree.ParseError as error:
            raise NormalizationError("invalid_xlsx_sheet_xml") from error
        rows = []
        for row_index, row in enumerate(root.iter(SHEET_NS + "row"), start=1):
            if row_index > MAX_XLSX_ROWS or total_cells >= MAX_XLSX_CELLS:
                warnings.append(f"xlsx_truncated:{sheet_name}")
                break
            values = []
            for cell in row.findall(SHEET_NS + "c"):
                if total_cells >= MAX_XLSX_CELLS:
                    warnings.append(f"xlsx_truncated:{sheet_name}")
                    break
                reference = cell.get("r") or "?"
                values.append(f"{reference}={_xlsx_cell_value(cell, shared_strings)}")
                total_cells += 1
            if values:
                rows.append(" | ".join(values))
        chunks.append(f"## Лист: {sheet_name}\n" + "\n".join(rows))
    return "\n\n".join(chunks), sorted(set(warnings))


def _office_text(payload: bytes, suffix: str) -> tuple[str, list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if suffix in {".docm", ".docx"}:
                return _docx_text(archive), []
            if suffix in {".pptm", ".pptx"}:
                return _pptx_text(archive), []
            return _xlsx_text(archive)
    except zipfile.BadZipFile as error:
        raise NormalizationError("invalid_ooxml_package") from error


def _pdf_text(payload: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="tof-context-pdf-") as directory_name:
        source = Path(directory_name) / "source.pdf"
        source.write_bytes(payload)
        try:
            result = subprocess.run(
                ["pdftotext", "-enc", "UTF-8", "-layout", str(source), "-"],
                text=True,
                capture_output=True,
                check=False,
                timeout=90,
            )
        except FileNotFoundError as error:
            raise NormalizationError("pdftotext_unavailable") from error
        except subprocess.TimeoutExpired as error:
            raise NormalizationError("pdftotext_timed_out") from error
    if result.returncode:
        raise NormalizationError("pdf_text_extraction_failed")
    if not result.stdout.strip():
        raise VisualReviewRequired("pdf_has_no_text_layer")
    pages = result.stdout.split("\f")
    # pdftotext commonly adds one terminal form-feed.  Keep intentional blank
    # pages while removing only that separator.
    if pages and not pages[-1].strip():
        pages.pop()
    return "\n\n".join(f"## Страница {number}\n{page.strip()}" for number, page in enumerate(pages, start=1))


def _normalized_text(payload: bytes, suffix: str) -> tuple[str, str, list[str]]:
    suffix = suffix.casefold()
    if suffix in TEXT_SUFFIXES:
        return _decode_text(payload), "text", []
    if suffix in HTML_SUFFIXES:
        return _html_text(payload), "html_text", []
    if suffix == ".json":
        return _json_text(payload), "json", []
    if suffix == ".csv":
        return _csv_text(payload), "csv_tsv", []
    if suffix in OFFICE_SUFFIXES:
        text, warnings = _office_text(payload, suffix)
        return text, "ooxml_text", warnings
    if suffix == ".pdf":
        return _pdf_text(payload), "pdf_text", []
    if suffix in VISUAL_SUFFIXES:
        raise VisualReviewRequired("visual_original_requires_agent_review")
    raise NormalizationError(f"unsupported_format:{suffix or '(no_extension)'}")


def _write_normalized(output_dir: Path, *, source_ref: str, payload: bytes, text: str, source_kind: str) -> Path:
    digest = hashlib.sha256(source_ref.encode("utf-8") + b"\0" + payload).hexdigest()
    target = output_dir / f"{digest[:20]}.txt"
    header = f"# Нормализованный локальный источник\nИсточник: {source_ref}\nФормат: {source_kind}\n\n"
    target.write_text(header + text.rstrip() + "\n", encoding="utf-8")
    return target


def _normalize_payload(
    *, channel: str, source_ref: str, payload: bytes, suffix: str, output_dir: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "channel": channel,
        "source_ref": source_ref,
        "source_sha256": _sha256_bytes(payload),
        "source_suffix": suffix.casefold(),
        "normalized_paths": [],
        "warnings": [],
    }
    try:
        text, source_kind, warnings = _normalized_text(payload, suffix)
        if not text.strip():
            raise VisualReviewRequired("source_has_no_readable_text")
        target = _write_normalized(
            output_dir, source_ref=source_ref, payload=payload, text=text, source_kind=source_kind
        )
        result.update(
            {
                "normalization_status": "normalized",
                "source_kind": source_kind,
                "normalized_paths": [str(target)],
                "warnings": warnings,
            }
        )
    except VisualReviewRequired as error:
        result.update(
            {
                "normalization_status": "visual_review_required",
                "reason": str(error),
                "review_path": source_ref,
            }
        )
    except NormalizationError as error:
        result.update({"normalization_status": "coverage_gap", "reason": str(error)})
    return result


def _normalize_archive(
    *, channel: str, source_ref: str, payload: bytes, output_dir: Path, depth: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "channel": channel,
        "source_ref": source_ref,
        "source_sha256": _sha256_bytes(payload),
        "source_suffix": ".zip",
        "source_kind": "zip_archive",
        "members": [],
        "normalized_paths": [],
        "warnings": [],
    }
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        result.update({"normalization_status": "coverage_gap", "reason": "invalid_zip_archive"})
        return result

    with archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) > MAX_ARCHIVE_MEMBERS:
            result["warnings"].append(f"archive_member_limit:{len(members)}")
            members = members[:MAX_ARCHIVE_MEMBERS]
        total_bytes = 0
        for member in members:
            member_ref = f"{source_ref}!{member.filename}"
            if member.flag_bits & 0x1:
                record = {
                    "channel": channel,
                    "source_ref": member_ref,
                    "source_suffix": Path(member.filename).suffix.casefold(),
                    "normalization_status": "coverage_gap",
                    "reason": "encrypted_zip_member",
                    "normalized_paths": [],
                    "warnings": [],
                }
            elif member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                record = {
                    "channel": channel,
                    "source_ref": member_ref,
                    "source_suffix": Path(member.filename).suffix.casefold(),
                    "normalization_status": "coverage_gap",
                    "reason": "archive_member_too_large",
                    "normalized_paths": [],
                    "warnings": [],
                }
            elif total_bytes + member.file_size > MAX_ARCHIVE_TOTAL_BYTES:
                record = {
                    "channel": channel,
                    "source_ref": member_ref,
                    "source_suffix": Path(member.filename).suffix.casefold(),
                    "normalization_status": "coverage_gap",
                    "reason": "archive_total_size_limit",
                    "normalized_paths": [],
                    "warnings": [],
                }
            else:
                try:
                    member_payload = archive.read(member)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    record = {
                        "channel": channel,
                        "source_ref": member_ref,
                        "source_suffix": Path(member.filename).suffix.casefold(),
                        "normalization_status": "coverage_gap",
                        "reason": "archive_member_unreadable",
                        "normalized_paths": [],
                        "warnings": [],
                    }
                else:
                    total_bytes += member.file_size
                    member_suffix = Path(member.filename).suffix.casefold()
                    if member_suffix == ".zip":
                        if depth >= MAX_ARCHIVE_DEPTH:
                            record = {
                                "channel": channel,
                                "source_ref": member_ref,
                                "source_suffix": member_suffix,
                                "normalization_status": "coverage_gap",
                                "reason": "nested_archive_depth_limit",
                                "normalized_paths": [],
                                "warnings": [],
                            }
                        else:
                            record = _normalize_archive(
                                channel=channel,
                                source_ref=member_ref,
                                payload=member_payload,
                                output_dir=output_dir,
                                depth=depth + 1,
                            )
                    else:
                        record = _normalize_payload(
                            channel=channel,
                            source_ref=member_ref,
                            payload=member_payload,
                            output_dir=output_dir,
                            suffix=member_suffix,
                        )
            result["members"].append(record)
            result["normalized_paths"].extend(record.get("normalized_paths", []))

    gaps = [member for member in result["members"] if member.get("normalization_status") != "normalized"]
    if result["normalized_paths"]:
        result["normalization_status"] = "normalized_with_gaps" if gaps else "normalized"
    else:
        result["normalization_status"] = "coverage_gap"
        if not result["members"]:
            result["reason"] = "zip_has_no_files"
    return result


def normalize_sources(sources: dict[str, Path], output_dir: Path) -> list[dict[str, Any]]:
    """Normalize supported local sources into text files for one claimed run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for channel, root in sorted(sources.items()):
        for path in _files(root):
            source_ref = str(path)
            try:
                size = path.stat().st_size
                if size > MAX_SOURCE_BYTES:
                    records.append(
                        {
                            "channel": channel,
                            "source_ref": source_ref,
                            "source_sha256": _sha256_file(path),
                            "source_suffix": path.suffix.casefold(),
                            "normalization_status": "coverage_gap",
                            "reason": "source_too_large",
                            "normalized_paths": [],
                            "warnings": [],
                        }
                    )
                    continue
                payload = path.read_bytes()
                if path.suffix.casefold() == ".zip":
                    records.append(
                        _normalize_archive(
                            channel=channel,
                            source_ref=source_ref,
                            payload=payload,
                            output_dir=output_dir,
                            depth=0,
                        )
                    )
                else:
                    records.append(
                        _normalize_payload(
                            channel=channel,
                            source_ref=source_ref,
                            payload=payload,
                            suffix=path.suffix,
                            output_dir=output_dir,
                        )
                    )
            except OSError as error:
                records.append(
                    {
                        "channel": channel,
                        "source_ref": source_ref,
                        "normalization_status": "coverage_gap",
                        "reason": f"source_unavailable:{type(error).__name__}",
                        "normalized_paths": [],
                        "warnings": [],
                    }
                )
    return records
