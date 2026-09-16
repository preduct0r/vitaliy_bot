from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import csv
import io
import zipfile

from scripts import context_normalization


def test_csv_multiline_quotes_after_sniffer_sample_preserve_records():
    rows = [
        ["id", "summary", "transcript"],
        ["1", "comma, forces quotes", "s" * 4100],
        ["2", "plain", 'one "quoted"\nnext line'],
    ]
    source = io.StringIO()
    csv.writer(source).writerows(rows)

    normalized = context_normalization._csv_text(source.getvalue().encode())

    assert list(csv.reader(io.StringIO(normalized), delimiter="\t")) == rows


def test_xlsx_package_absolute_sheet_target_is_read_without_external_links():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Review" sheetId="1" r:id="rId1"/>'
            '<sheet name="External" sheetId="2" r:id="rId2"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="/xl/worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml" TargetMode="External"/>'
            '</Relationships>',
        )
        for number in (1, 2):
            archive.writestr(
                f"xl/worksheets/sheet{number}.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetData><row r="1"><c r="A1" t="inlineStr"><is>'
                f'<t>text {number}</t></is></c></row></sheetData></worksheet>',
            )
    with zipfile.ZipFile(payload) as archive:
        text, warnings = context_normalization._xlsx_text(archive)

    assert "A1=text 1" in text
    assert "text 2" not in text
    assert warnings == []


def _member(record: dict, suffix: str) -> dict:
    return next(item for item in record["members"] if item["source_ref"].endswith(suffix))


def test_telegram_zip_becomes_local_text_views_without_extracting_members(tmp_path: Path):
    export = tmp_path / "telegram_export" / "telegram.zip"
    export.parent.mkdir(parents=True)
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("messages.html", "<p>Денис: проверь входные данные.</p>")
        archive.writestr("result.json", '{"messages":[{"text":"Не терять уточнения"}]}')
        archive.writestr("../../not-an-extracted-file.txt", "Путь не должен быть извлечён")
        archive.writestr("media/photo.jpg", b"\xff\xd8")
    original = export.read_bytes()
    normalized_dir = tmp_path / "normalized"

    record = context_normalization.normalize_sources(
        {"telegram_export": export.parent}, normalized_dir
    )[0]

    assert record["normalization_status"] == "normalized_with_gaps"
    html = _member(record, "!messages.html")
    json_record = _member(record, "!result.json")
    unsafe = _member(record, "!../../not-an-extracted-file.txt")
    photo = _member(record, "!media/photo.jpg")
    assert "Денис: проверь входные данные." in Path(html["normalized_paths"][0]).read_text(encoding="utf-8")
    assert "Не терять уточнения" in Path(json_record["normalized_paths"][0]).read_text(encoding="utf-8")
    assert "Путь не должен быть извлечён" in Path(unsafe["normalized_paths"][0]).read_text(encoding="utf-8")
    assert all(path.parent == normalized_dir for path in map(Path, record["normalized_paths"]))
    assert photo["normalization_status"] == "visual_review_required"
    assert export.read_bytes() == original
    assert not (tmp_path / "not-an-extracted-file.txt").exists()


def test_ooxml_exports_become_readable_local_text(tmp_path: Path):
    docx = tmp_path / "telegram_export" / "note.docx"
    docx.parent.mkdir(parents=True)
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Правило из вложения</w:t></w:r></w:p></w:body></w:document>",
        )
    xlsx = tmp_path / "telegram_export" / "table.xlsx"
    with zipfile.ZipFile(xlsx, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Чат" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<si><t>Пользователь</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="inlineStr"><is><t>Добавить правило</t></is></c></row></sheetData></worksheet>',
        )

    records = context_normalization.normalize_sources(
        {"telegram_export": docx.parent}, tmp_path / "normalized"
    )
    by_suffix = {record["source_suffix"]: record for record in records}

    assert "Правило из вложения" in Path(by_suffix[".docx"]["normalized_paths"][0]).read_text(encoding="utf-8")
    xlsx_text = Path(by_suffix[".xlsx"]["normalized_paths"][0]).read_text(encoding="utf-8")
    assert "## Лист: Чат" in xlsx_text
    assert "A1=Пользователь" in xlsx_text
    assert "B1=Добавить правило" in xlsx_text


def test_one_nested_telegram_zip_is_normalized_and_corrupt_zip_is_visible(tmp_path: Path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("messages.txt", "Уточнение из вложенного архива")
    outer = tmp_path / "telegram_export" / "outer.zip"
    outer.parent.mkdir(parents=True)
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("export.zip", inner.getvalue())
    corrupt = tmp_path / "telegram_export" / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")

    records = context_normalization.normalize_sources(
        {"telegram_export": outer.parent}, tmp_path / "normalized"
    )
    by_name = {Path(record["source_ref"]).name: record for record in records}
    nested = by_name["outer.zip"]

    assert nested["normalization_status"] == "normalized"
    nested_member = nested["members"][0]
    assert "Уточнение из вложенного архива" in Path(nested_member["normalized_paths"][0]).read_text(encoding="utf-8")
    assert by_name["corrupt.zip"]["normalization_status"] == "coverage_gap"
    assert by_name["corrupt.zip"]["reason"] == "invalid_zip_archive"


def test_pdf_and_unsupported_telegram_files_remain_explicit(monkeypatch, tmp_path: Path):
    pdf = tmp_path / "telegram_export" / "message.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7\n")
    movie = tmp_path / "telegram_export" / "voice.mp4"
    movie.write_bytes(b"not-a-transcript")
    transcript = tmp_path / "telegram_export" / "voice.vtt"
    transcript.write_text("WEBVTT\n\n00:00.000 --> 00:02.000\nТекст голосового сообщения\n", encoding="utf-8")

    monkeypatch.setattr(
        context_normalization.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="Текст из PDF\n", stderr=""),
    )
    records = context_normalization.normalize_sources(
        {"telegram_export": pdf.parent}, tmp_path / "normalized"
    )
    by_suffix = {record["source_suffix"]: record for record in records}

    assert "Текст из PDF" in Path(by_suffix[".pdf"]["normalized_paths"][0]).read_text(encoding="utf-8")
    assert "Текст голосового сообщения" in Path(by_suffix[".vtt"]["normalized_paths"][0]).read_text(encoding="utf-8")
    assert by_suffix[".mp4"]["normalization_status"] == "coverage_gap"
    assert by_suffix[".mp4"]["reason"] == "unsupported_format:.mp4"
