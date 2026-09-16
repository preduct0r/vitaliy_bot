"""Validate, normalize, and visually prepare Telegram document uploads."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scripts import artifact_qa, context_normalization

from .models import ArtifactInput, InputError, PreparedArtifact
from .storage import VitalyStore


SUPPORTED_SUFFIXES = {".pptx", ".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv", ".png", ".jpeg", ".jpg"}
MEETING_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_CASE_ARTIFACTS = 10
MAX_CASE_UNITS = 200
MAX_CONTEXT_CHARS = 70_000
MAX_MODEL_IMAGES = 12


def safe_filename(value: str) -> str:
    name = Path(value or "document").name
    name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._ -]+", "_", name).strip(" .")
    return name[:120] or "document"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ooxml_unit_count(path: Path, prefix: str, pattern: re.Pattern[str]) -> int:
    try:
        with zipfile.ZipFile(path) as archive:
            return sum(1 for name in archive.namelist() if pattern.fullmatch(name))
    except zipfile.BadZipFile:
        return 0


def preflight_unit_count(path: Path) -> int:
    suffix = path.suffix.casefold()
    if suffix == ".pptx":
        return _ooxml_unit_count(path, "ppt/slides", re.compile(r"ppt/slides/slide\d+\.xml")) or 1
    if suffix == ".xlsx":
        return _ooxml_unit_count(path, "xl/worksheets", re.compile(r"xl/worksheets/sheet\d+\.xml")) or 1
    if suffix == ".pdf":
        try:
            completed = subprocess.run(
                ["pdfinfo", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
        except OSError:
            return 1
        if completed.returncode == 0:
            match = re.search(r"^Pages:\s*(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
            if match:
                return int(match.group(1))
    return 1


def _select_images(paths: Sequence[Path]) -> List[Path]:
    existing = [path for path in paths if path.exists()]
    if len(existing) <= MAX_MODEL_IMAGES:
        return existing
    step = (len(existing) - 1) / float(MAX_MODEL_IMAGES - 1)
    return [existing[round(index * step)] for index in range(MAX_MODEL_IMAGES)]


def _normalized_body(value: str) -> str:
    """Drop normalizer provenance headers; the model needs material, not Mac paths."""

    if value.startswith("# Нормализованный локальный источник\n"):
        _header, separator, body = value.partition("\n\n")
        if separator:
            return body
    return value


class DocumentProcessor:
    """Prepares one file while leaving the original upload unchanged."""

    def __init__(self, store: VitalyStore) -> None:
        self.store = store

    def validate_input(self, case_id: str, artifact: ArtifactInput) -> Tuple[str, int]:
        suffix = Path(artifact.original_name).suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise InputError("Поддерживаются PPTX, PDF, DOCX, XLSX, TXT, MD, CSV, PNG и JPEG")
        if artifact.size_bytes > MAX_FILE_BYTES:
            raise InputError("Один файл не должен быть больше 50 МБ")
        if self.store.artifact_count(case_id) >= MAX_CASE_ARTIFACTS:
            raise InputError("В одном кейсе можно использовать не более 10 файлов")
        units = preflight_unit_count(artifact.path)
        if self.store.total_units(case_id) + units > MAX_CASE_UNITS:
            raise InputError("Суммарный лимит кейса — 200 страниц, слайдов или листов")
        return suffix, units

    def ingest(self, case_id: str, artifact: ArtifactInput) -> Dict[str, object]:
        suffix, estimated_units = self.validate_input(case_id, artifact)
        case_dir = self.store.case_dir(case_id)
        artifacts_dir = case_dir / "artifacts"
        name = safe_filename(artifact.original_name)
        target = artifacts_dir / (uuid_safe_prefix(artifact.telegram_file_id) + "_" + name)
        if artifact.path.resolve() != target.resolve():
            shutil.copy2(artifact.path, target)
        normalized_dir = case_dir / "normalized" / uuid_safe_prefix(artifact.telegram_file_id)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        records = context_normalization.normalize_sources({"telegram_upload": target}, normalized_dir)
        record = records[0]
        warnings = list(record.get("warnings", []))
        normalized_paths = [Path(item) for item in record.get("normalized_paths", [])]
        normalized_path: Optional[Path] = normalized_paths[0] if normalized_paths else None

        image_paths: List[Path] = []
        rendered_dir = case_dir / "rendered" / uuid_safe_prefix(artifact.telegram_file_id)
        if suffix in {".pptx", ".pdf", ".docx", ".xlsx", ".csv"}:
            try:
                rendered = artifact_qa.render_file(target, rendered_dir)
                image_paths = [Path(page["path"]) for page in rendered.get("pages", [])]
            except (artifact_qa.ArtifactQaError, OSError, RuntimeError) as error:
                warnings.append("render_failed:" + type(error).__name__)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            image_paths = [target]

        actual_units = len(image_paths) if image_paths else estimated_units
        if self.store.total_units(case_id) + actual_units > MAX_CASE_UNITS:
            shutil.rmtree(rendered_dir, ignore_errors=True)
            raise InputError("Суммарный лимит кейса — 200 страниц, слайдов или листов")
        return self.store.add_artifact(
            case_id=case_id,
            telegram_file_id=artifact.telegram_file_id,
            original_name=artifact.original_name,
            stored_name=target.name,
            sha256=sha256(target),
            suffix=suffix,
            size_bytes=artifact.size_bytes,
            local_path=target,
            normalized_text_path=normalized_path,
            rendered_pages=image_paths,
            unit_count=actual_units,
            warnings=warnings,
        )

    def prepared(self, case_id: str) -> List[PreparedArtifact]:
        result: List[PreparedArtifact] = []
        for artifact in self.store.artifacts(case_id):
            normalized_path = artifact.get("normalized_text_path")
            text = ""
            if normalized_path and Path(normalized_path).exists():
                text = _normalized_body(Path(normalized_path).read_text(encoding="utf-8", errors="replace"))
            if not text:
                text = "[Текстовый слой отсутствует; см. приложенные изображения.]"
            result.append(
                PreparedArtifact(
                    artifact_id=str(artifact["id"]),
                    name=str(artifact["original_name"]),
                    text=text,
                    text_reference="Файл: " + str(artifact["original_name"]),
                    image_paths=_select_images([Path(path) for path in artifact["rendered_pages"]]),
                    unit_count=int(artifact["unit_count"]),
                    warnings=list(artifact["warnings"]),
                )
            )
        return result

    def material_chunks(self, case_id: str) -> List[str]:
        chunks: List[str] = []
        for artifact in self.prepared(case_id):
            prefix = "\n\n=== " + artifact.text_reference + " ===\n"
            text = artifact.text
            if len(text) <= MAX_CONTEXT_CHARS:
                chunks.append(prefix + text)
                continue
            for offset in range(0, len(text), MAX_CONTEXT_CHARS):
                chunks.append(prefix + "Фрагмент " + str(offset // MAX_CONTEXT_CHARS + 1) + "\n" + text[offset : offset + MAX_CONTEXT_CHARS])
        return chunks

    def selected_images(self, case_id: str) -> List[Path]:
        selected: List[Path] = []
        for artifact in self.prepared(case_id):
            selected.extend(artifact.image_paths)
        return _select_images(selected)

    def prepare_meeting_source(
        self, submission_id: str, artifact: ArtifactInput, *, min_free_bytes: int = 0
    ) -> Dict[str, object]:
        """Archive a transcript without turning it into a user's case artifact.

        Meeting uploads deliberately use the same normalizer and renderer as a
        review, but live under the external knowledge archive.  A scanned PDF
        with no text is rejected because speaker attribution cannot be audited.
        """

        suffix = Path(artifact.original_name).suffix.casefold()
        if suffix not in MEETING_SUFFIXES:
            raise InputError("Для встречи поддерживаются только TXT, MD, DOCX и текстовый PDF")
        if artifact.size_bytes > MAX_FILE_BYTES:
            raise InputError("Один файл не должен быть больше 50 МБ")
        if preflight_unit_count(artifact.path) > MAX_CASE_UNITS:
            raise InputError("Транскрипт не должен содержать больше 200 страниц")
        self.store.ensure_free_space(min_free_bytes)
        root = self.store.knowledge_root / "meetings" / submission_id
        source_dir = root / "source"
        normalized_dir = root / "normalized"
        rendered_dir = root / "rendered"
        source_dir.mkdir(parents=True, exist_ok=True)
        target = source_dir / safe_filename(artifact.original_name)
        if artifact.path.resolve() != target.resolve():
            shutil.copy2(artifact.path, target)
        records = context_normalization.normalize_sources({"meeting_transcript": target}, normalized_dir)
        record = records[0]
        normalized_paths = [Path(item) for item in record.get("normalized_paths", [])]
        if not normalized_paths or not normalized_paths[0].exists():
            raise InputError("Не удалось извлечь текст транскрипта")
        normalized_path = normalized_paths[0]
        body = _normalized_body(normalized_path.read_text(encoding="utf-8", errors="replace")).strip()
        if not body or body.startswith("[Текстовый слой отсутствует"):
            raise InputError("В PDF нет доступного текстового слоя. Загрузите текстовый PDF или TXT/DOCX.")
        try:
            if suffix in {".pdf", ".docx"}:
                artifact_qa.render_file(target, rendered_dir)
        except (artifact_qa.ArtifactQaError, OSError, RuntimeError):
            # Render is retained when available, but extracted text is enough to
            # classify a transcript safely.
            pass
        return {
            "source_name": target.name,
            "source_path": target,
            "normalized_path": normalized_path,
            "source_sha256": sha256(target),
            "normalized_sha256": hashlib.sha256(_semantic_text(body).encode("utf-8")).hexdigest(),
            "text": body,
        }


def uuid_safe_prefix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _semantic_text(value: str) -> str:
    """Stable duplicate signal that tolerates harmless whitespace changes."""

    return re.sub(r"\s+", " ", value.casefold()).strip()
