"""Типизированные загрузчики таблиц ledger.

Точка входа — `load_table(name)` и именованные обёртки (`load_work_orders` и т.д.).
Каталог ledger по умолчанию берётся из `config.resolve_data_parser_ledger_dir`
(`TOF_DATA_PARSER_LEDGER_DIR`), см. `DATA_MAP.md`, слой 2.

Что даёт загрузчик поверх `pd.read_csv`:

- проверку обязательных колонок из `schemas.py` — их отсутствие это ошибка
  `LedgerSchemaError` с именем таблицы, путём и списком недостающих колонок;
- выбор носителя: parquet, если он есть рядом и не помечен устаревшим
  (`parquet_stale/`), иначе CSV;
- одинаковый тип данных: все колонки — строки, пустое значение — `""`;
- фильтры `year` / `period` по датам таблицы и выбор колонок `columns`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from tof_reliability.data.schemas import LEDGER_TABLES, LedgerTableSchema, table_schema


class LedgerSchemaError(ValueError):
    """В файле таблицы нет колонок, без которых её нельзя использовать."""


class LedgerTableNotFoundError(FileNotFoundError):
    """Файла таблицы нет в каталоге ledger."""


PARQUET_STALE_DIR = "parquet_stale"
E2E_CHUNK_ROWS = 100_000

# The generic ``event_id`` is an event identity, not a universal row key.
# ``source_records`` deliberately has several provenance records for one event;
# treating that relationship as a duplicate hides the real grain from E2E.
# Parser issues are instead identified by their full reproducible message at a
# source, because an issue code itself may recur for several documents.
IDENTITY_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "source_records": (),
    "parser_issues": ("issue_id", "source_file", "severity", "message"),
}


def resolve_ledger_dir(ledger_dir: str | Path | None = None) -> Path:
    """Каталог ledger: явный аргумент, иначе `TOF_DATA_PARSER_LEDGER_DIR` / дефолт."""

    if ledger_dir is not None:
        return Path(ledger_dir).expanduser()
    from tof_reliability.config import resolve_data_parser_ledger_dir

    repo_root = Path(__file__).resolve().parents[3]
    return resolve_data_parser_ledger_dir(repo_root)


def _parquet_is_stale(ledger_dir: Path, name: str) -> bool:
    """Parquet считается устаревшим, если его двойник лежит в `parquet_stale/`.

    `data_parser` при инкрементальном рефреше переносит затронутые зеркала в этот
    каталог и оставляет CSV единственным актуальным носителем.
    """

    stale_dir = ledger_dir / PARQUET_STALE_DIR
    if not stale_dir.is_dir():
        return False
    return any(path.name.startswith(f"{name}.") for path in stale_dir.glob(f"{name}.*.parquet"))


def _parquet_engine_available() -> bool:
    try:  # pandas сам решает, каким движком читать; без движка parquet недоступен
        pd.io.parquet.get_engine("auto")
    except Exception:
        return False
    return True


def table_path(name: str, ledger_dir: str | Path | None = None) -> Path | None:
    """Файл таблицы: parquet, если он есть и не устарел, иначе CSV; иначе `None`."""

    schema = table_schema(name)
    directory = resolve_ledger_dir(ledger_dir)
    parquet = directory / f"{schema.name}.parquet"
    if parquet.exists() and not _parquet_is_stale(directory, schema.name) and _parquet_engine_available():
        return parquet
    csv = directory / f"{schema.name}.csv"
    if csv.exists():
        return csv
    return None


def _as_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Все колонки — строки, пропуски — пустая строка (контракт ledger)."""

    if frame.empty:
        return frame.astype(str) if len(frame.columns) else frame
    return frame.fillna("").astype(str)


def _available_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return list(pd.read_parquet(path).columns)
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        raise LedgerSchemaError(
            f"{path} пуст: нет даже строки заголовка. Пустая таблица ledger — это сломанный файл, "
            "а не «нет данных»: пересоберите ledger в data_parser"
        ) from None


def _read_frame(path: Path, usecols: list[str] | None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=usecols) if usecols else pd.read_parquet(path)
        return _as_text_frame(frame)
    if usecols is not None:
        return pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, low_memory=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def _period_series(frame: pd.DataFrame, schema: LedgerTableSchema) -> pd.Series:
    """Период строки `YYYY-MM` по первой непустой дате из `schema.date_columns`."""

    if not schema.date_columns:
        raise LedgerSchemaError(
            f"таблица {schema.name} не имеет колонок с датой, фильтры year/period к ней неприменимы"
        )
    present = [column for column in schema.date_columns if column in frame.columns]
    if not present:
        raise LedgerSchemaError(
            f"в {schema.name} нет ни одной колонки с датой ({', '.join(schema.date_columns)}), "
            "фильтры year/period применить не к чему"
        )
    raw = pd.Series([""] * len(frame), index=frame.index, dtype="object")
    for column in present:
        values = frame[column].astype(str).str.strip()
        raw = raw.where(raw != "", values)
    parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    return parsed.dt.strftime("%Y-%m").fillna("")


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (str(value),)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _filter_period(frame: pd.DataFrame, schema: LedgerTableSchema, year: object, period: object) -> pd.DataFrame:
    years = _as_tuple(year)
    periods = _as_tuple(period)
    if not years and not periods:
        return frame
    for value in periods:
        if len(value) not in {4, 7}:
            raise ValueError(f"period должен быть 'YYYY' или 'YYYY-MM', получено {value!r}")
    stamps = _period_series(frame, schema)
    keep = pd.Series(False, index=frame.index)
    for value in years:
        keep |= stamps.str.startswith(f"{value}-")
    for value in periods:
        keep |= stamps.str.startswith(value) if len(value) == 4 else stamps.eq(value)
    return frame[keep]


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def load_table(
    name: str,
    *,
    ledger_dir: str | Path | None = None,
    columns: Sequence[str] | None = None,
    year: int | Iterable[int] | None = None,
    period: str | Iterable[str] | None = None,
    missing_ok: bool = False,
) -> pd.DataFrame:
    """Прочитать таблицу ledger по имени.

    `columns` — какие колонки вернуть и в каком порядке; отсутствующие в файле
    необязательные колонки возвращаются пустыми, обязательные по схеме — ошибка.
    `year` — год или несколько лет, `period` — `YYYY` или `YYYY-MM` (можно список);
    период строки считается по датам таблицы (`schemas.date_columns`).
    `missing_ok=True` — отсутствующий файл даёт пустую таблицу вместо ошибки.
    """

    schema = table_schema(name)
    directory = resolve_ledger_dir(ledger_dir)
    path = table_path(schema.name, directory)
    requested = list(columns) if columns is not None else None
    if path is None:
        if missing_ok:
            return _empty_frame(requested if requested is not None else schema.required_columns)
        raise LedgerTableNotFoundError(
            f"таблицы {schema.name} нет в каталоге ledger {directory}; "
            "проверьте TOF_DATA_PARSER_LEDGER_DIR или пересоберите ledger в data_parser"
        )

    available = _available_columns(path)
    missing_required = [column for column in schema.required_columns if column not in available]
    if missing_required:
        raise LedgerSchemaError(
            f"{schema.name}: в {path} нет обязательных колонок {', '.join(missing_required)}; "
            f"контракт таблицы — {schema.contract}"
        )

    usecols: list[str] | None = None
    if requested is not None:
        needed = list(dict.fromkeys([*requested, *schema.date_columns])) if (year or period) else requested
        usecols = [column for column in dict.fromkeys(needed) if column in available]
    frame = _read_frame(path, usecols)
    frame = _filter_period(frame, schema, year, period)
    if requested is not None:
        for column in requested:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[requested]
    return frame.reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_columns(schema: LedgerTableSchema, requested: Sequence[str]) -> tuple[str, ...]:
    configured = IDENTITY_COLUMNS_BY_TABLE.get(schema.name)
    if configured is not None:
        return tuple(column for column in configured if column in requested)
    for column in ("event_id", "source_document_id", "criticality_record_id"):
        if column in requested:
            return (column,)
    return ()


def _identity_values(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if len(columns) == 1:
        return frame[str(columns[0])].astype(str).str.strip()
    return frame.loc[:, list(columns)].fillna("").astype(str).agg("\x1f".join, axis=1).str.strip()


def summarize_table(
    name: str,
    *,
    ledger_dir: str | Path | None = None,
    join_keys: Sequence[str] = (),
    chunk_rows: int = E2E_CHUNK_ROWS,
    collect_join_values: bool = False,
) -> dict[str, object]:
    """Read a ledger table through its contract without retaining all rows.

    This is deliberately narrower than :func:`load_table`: it is for a
    server-side system check that needs row counts, coverage and one carrier
    checksum. CSV is processed in bounded chunks; event identity duplicates
    are detected across chunks.  The analytical result itself must still use
    ``load_table`` or a named typed loader.
    """

    if chunk_rows < 1:
        raise ValueError("chunk_rows должен быть положительным")
    schema = table_schema(name)
    directory = resolve_ledger_dir(ledger_dir)
    path = table_path(schema.name, directory)
    if path is None:
        raise LedgerTableNotFoundError(
            f"таблицы {schema.name} нет в каталоге ledger {directory}; "
            "проверьте TOF_DATA_PARSER_LEDGER_DIR или пересоберите ledger в data_parser"
        )
    available = _available_columns(path)
    missing_required = [column for column in schema.required_columns if column not in available]
    if missing_required:
        raise LedgerSchemaError(
            f"{schema.name}: в {path} нет обязательных колонок {', '.join(missing_required)}; "
            f"контракт таблицы — {schema.contract}"
        )
    requested = list(dict.fromkeys([*schema.required_columns, *[str(key) for key in join_keys if str(key) in available]]))
    identity_columns = _identity_columns(schema, requested)
    identity = " + ".join(identity_columns)
    rows_read = 0
    identity_duplicates = 0
    identities: set[str] = set()
    key_counts = {key: 0 for key in join_keys if key in requested}
    key_values = {key: set() for key in key_counts} if collect_join_values else {}
    if path.suffix == ".csv":
        reader = pd.read_csv(
            path,
            usecols=requested,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            chunksize=chunk_rows,
        )
        for frame in reader:
            rows_read += len(frame)
            if identity_columns:
                values = _identity_values(frame, identity_columns)
                for value in values[values != ""]:
                    if value in identities:
                        identity_duplicates += 1
                    else:
                        identities.add(value)
            for key in key_counts:
                values = frame[key].astype(str).str.strip()
                nonempty = values[values != ""]
                key_counts[key] += int(len(nonempty))
                if collect_join_values:
                    key_values[key].update(nonempty.tolist())
    else:
        # Published TOF generations currently carry CSV mirrors.  Retain a
        # correct fallback for a future parquet-only generation rather than
        # silently ignoring it; its use is visible in the evidence summary.
        frame = load_table(schema.name, ledger_dir=directory, columns=requested)
        rows_read = len(frame)
        if identity_columns:
            identity_duplicates = int(_identity_values(frame, identity_columns).duplicated(keep=False).sum())
        for key in key_counts:
            values = frame[key].astype(str).str.strip()
            nonempty = values[values != ""]
            key_counts[key] = int(len(nonempty))
            if collect_join_values:
                key_values[key].update(nonempty.tolist())
    return {
        "table": schema.name,
        "path": str(path),
        "carrier": path.suffix.lstrip("."),
        "rows_read": int(rows_read),
        "columns": requested,
        "identity_column": identity,
        "identity_duplicate_rows": int(identity_duplicates) if identity_columns else None,
        "deduplication_status": "pass" if identity_columns and not identity_duplicates else "fail" if identity_columns else "not_applicable",
        "key_counts": key_counts,
        "content_sha256": _file_sha256(path),
        "_join_values": key_values,
    }


def load_work_orders(**kwargs) -> pd.DataFrame:
    """Заказы ТОРО (`work_order_events`). Единица счёта ремонта — эпизод (`R-CNT-001`)."""

    return load_table("work_order_events", **kwargs)


def load_maintenance_plan_events(**kwargs) -> pd.DataFrame:
    """Позиции месячных и годовых графиков (`maintenance_plan_events`)."""

    return load_table("maintenance_plan_events", **kwargs)


def load_downtime_events(**kwargs) -> pd.DataFrame:
    """Простои (`downtime_events`)."""

    return load_table("downtime_events", **kwargs)


def load_mtr_writeoffs(**kwargs) -> pd.DataFrame:
    """Списания МТР (`mtr_writeoff_events`). Источник — нижняя оценка (`R-CST-005`)."""

    return load_table("mtr_writeoff_events", **kwargs)


def load_work_order_operations(**kwargs) -> pd.DataFrame:
    """Операции заказов SAP и часы (`work_order_operation_events`)."""

    return load_table("work_order_operation_events", **kwargs)


def load_equipment_criticality(**kwargs) -> pd.DataFrame:
    """Критичность оборудования (`equipment_criticality_records`)."""

    return load_table("equipment_criticality_records", **kwargs)


def load_tech_card_headers(**kwargs) -> pd.DataFrame:
    """Заголовки ТК (`tech_card_headers`): точный ключ ЕО + счётчик, стратегия и нормативы."""

    return load_table("tech_card_headers", **kwargs)


def load_tech_card_operations(**kwargs) -> pd.DataFrame:
    """Операции техкарт (`tech_card_operations`)."""

    return load_table("tech_card_operations", **kwargs)


__all__ = [
    "LEDGER_TABLES",
    "LedgerSchemaError",
    "LedgerTableNotFoundError",
    "load_downtime_events",
    "load_equipment_criticality",
    "load_maintenance_plan_events",
    "load_mtr_writeoffs",
    "load_work_order_operations",
    "load_table",
    "summarize_table",
    "load_tech_card_headers",
    "load_tech_card_operations",
    "load_work_orders",
    "resolve_ledger_dir",
    "table_path",
]
