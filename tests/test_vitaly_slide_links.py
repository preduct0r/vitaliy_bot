from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.context_normalization import _pptx_text


def _audit():
    return {
        "id": "audit123456789012",
        "revision": 0,
        "current_json": json.dumps({
            "critical_gaps": [{
                "title": "Нет расчёта",
                "detail": "Не показана формула",
                "material_reference": "план.pptx",
                "material_locations": [{"file": "план.pptx", "kind": "slide", "number": 4}],
            }],
            "contradictions": [{
                "title": "Разные пики",
                "detail": "Числа расходятся",
                "material_reference": "план.pptx",
                "material_locations": [
                    {"file": "план.pptx", "kind": "slide", "number": 4},
                    {"file": "план.pptx", "kind": "slide", "number": 7},
                ],
            }],
            "questions": [],
        }),
    }


def test_format_review_prints_both_slide_numbers():
    pytest.importorskip("aiogram")
    from tof_reliability.vitaly_bot.telegram import format_review
    text = format_review({
        "readiness_score": 50,
        "verdict": "Нужно уточнить",
        "critical_gaps": [],
        "contradictions": [_audit_item for _audit_item in json.loads(_audit()["current_json"])["contradictions"]],
        "questions": [],
        "prep_checklist": [],
    })
    assert "C1. Слайды 4 и 7 (план.pptx): Разные пики" in text


def test_slide_keyboard_uses_short_callbacks():
    pytest.importorskip("aiogram")
    from tof_reliability.vitaly_bot.correction_ui import slide_keyboard
    markup = slide_keyboard(_audit(), 0)
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert any(button.callback_data.endswith("locpick:C1") for button in buttons)
    assert all(len(button.callback_data.encode()) <= 64 for button in buttons)


def test_pptx_text_follows_presentation_order(tmp_path: Path):
    path = tmp_path / "ordered.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/presentation.xml",
            '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<p:sldIdLst><p:sldId id="1" r:id="rId2"/><p:sldId id="2" r:id="rId1"/></p:sldIdLst></p:presentation>',
        )
        archive.writestr(
            "ppt/_rels/presentation.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="slides/slide1.xml" Type="slide"/>'
            '<Relationship Id="rId2" Target="slides/slide2.xml" Type="slide"/></Relationships>',
        )
        for number, text in ((1, "ВТОРОЙ"), (2, "ПЕРВЫЙ")):
            archive.writestr(
                f"ppt/slides/slide{number}.xml",
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:t>' + text + '</a:t></p:sld>',
            )
    with zipfile.ZipFile(path) as archive:
        text = _pptx_text(archive)
    assert text.index("## Слайд 1") < text.index("## Слайд 2")
    assert text.index("ПЕРВЫЙ") < text.index("ВТОРОЙ")
