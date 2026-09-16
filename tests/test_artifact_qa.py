import json
import zipfile
from pathlib import Path

from scripts import artifact_qa


def _ooxml(path: Path, required_part: str, *extra_parts: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(required_part, "<document/>")
        for part in extra_parts:
            archive.writestr(part, "<part/>")
    return path


def test_inspect_recognizes_all_office_formats_and_cyrillic_csv(tmp_path):
    xlsx = _ooxml(tmp_path / "книга.xlsx", "xl/workbook.xml")
    docx = _ooxml(tmp_path / "документ.docx", "word/document.xml")
    pptx = _ooxml(tmp_path / "слайды.pptx", "ppt/presentation.xml")
    pdf = tmp_path / "файл.pdf"
    pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n")
    csv = tmp_path / "данные.csv"
    csv.write_text("Наименование,Значение\nНасос,42\n", encoding="utf-8")

    assert artifact_qa.inspect_file(xlsx)["status"] == "passing"
    assert artifact_qa.inspect_file(docx)["format"] == "docx"
    assert artifact_qa.inspect_file(pptx)["format"] == "pptx"
    assert artifact_qa.inspect_file(pdf)["format"] == "pdf"
    assert artifact_qa.inspect_file(csv)["status"] == "passing"


def test_inspect_blocks_complex_office_features_without_editing_source(tmp_path):
    source = _ooxml(
        tmp_path / "macro.xlsm",
        "xl/workbook.xml",
        "xl/vbaProject.bin",
        "xl/externalLinks/externalLink1.xml",
        "xl/pivotTables/pivotTable1.xml",
    )
    before = source.read_bytes()

    report = artifact_qa.inspect_file(source)

    assert report["status"] == "blocked"
    assert {"vba", "external_links", "pivot_tables"} <= set(report["features"])
    assert source.read_bytes() == before


def test_inspect_reports_corrupt_ooxml_as_failed(tmp_path):
    source = tmp_path / "broken.docx"
    source.write_bytes(b"not a zip")

    report = artifact_qa.inspect_file(source)

    assert report["status"] == "failed"
    assert report["checks"][0]["status"] == "failed"


def test_verify_writes_stable_machine_report_with_rendered_pages(tmp_path, monkeypatch):
    source = _ooxml(tmp_path / "книга.xlsx", "xl/workbook.xml")
    output = tmp_path / "qa"

    monkeypatch.setattr(
        artifact_qa,
        "render_file",
        lambda _source, _output: {
            "status": "passing",
            "pages": [{"number": 1, "path": str(output / "page-1.png"), "bytes": 10, "width": 100, "height": 200}],
        },
    )
    monkeypatch.setattr(artifact_qa, "_tool_version", lambda _command: "test-version")

    report = artifact_qa.verify_file(source, output)
    saved = json.loads((output / "artifact_qa.json").read_text(encoding="utf-8"))

    assert report["status"] == "passing"
    assert saved["artifact"]["sha256"] == artifact_qa._sha256(source)
    assert saved["pages"][0]["number"] == 1
    assert saved["tool_versions"]["libreoffice"] == "test-version"


def test_verify_preserves_blocked_status_after_successful_render(tmp_path, monkeypatch):
    source = _ooxml(tmp_path / "macro.xlsm", "xl/workbook.xml", "xl/vbaProject.bin")
    monkeypatch.setattr(artifact_qa, "render_file", lambda _source, _output: {"pages": []})

    report = artifact_qa.verify_file(source, tmp_path / "qa")

    assert report["status"] == "blocked"
    assert report["blockers"]
