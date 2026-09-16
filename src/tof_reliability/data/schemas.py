"""Схемы таблиц ledger: что обязано быть в файле, чтобы им можно было пользоваться.

Одна таблица — одна `LedgerTableSchema`. Обязательные колонки — это контракт
`data_parser` (`DATA_MAP.md`, слой 1), а не полный список колонок: полная схема
события шире и меняется чаще. Загрузчик падает, если обязательной колонки нет,
поэтому «тихо посчитали по пустому полю» становится невозможным.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LedgerTableSchema:
    """Контракт одной таблицы ledger.

    `required_columns` — минимальный набор, без которого таблица не отвечает на
    свой вопрос. `date_columns` — колонки, по которым считается период строки:
    берётся первое непустое значение в указанном порядке. `rules` — правила
    реестра, применимые к таблице. `contract` — файл контракта в `docs/contracts/`.
    """

    name: str
    title: str
    required_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()

    @property
    def contract(self) -> str:
        return f"docs/contracts/{self.name}.md"


_SCHEMAS: tuple[LedgerTableSchema, ...] = (
    LedgerTableSchema(
        name="work_order_events",
        title="SAP-заказы ТОРО: факт ремонтных воздействий",
        required_columns=("event_id", "source_file", "order_id", "equipment_id", "work_type_raw", "event_date"),
        date_columns=("event_date", "actual_start", "planned_start"),
        rules=("R-CNT-001", "R-CNT-002", "R-CNT-005", "R-CST-001", "R-CST-006"),
    ),
    LedgerTableSchema(
        name="maintenance_plan_events",
        title="Позиции месячных и годовых графиков",
        required_columns=("event_id", "source_file", "source_sheet", "equipment_id", "work_type_raw", "planned_start"),
        date_columns=("planned_start", "event_date"),
        rules=("R-CNT-003", "R-CNT-006", "R-CNT-007", "R-CNT-010"),
    ),
    LedgerTableSchema(
        name="maintenance_fact_events",
        title="Факт ТОиР из выгрузок «Выгрузка ТОФ *»",
        required_columns=("event_id", "source_file", "equipment_id", "work_type_raw", "event_date"),
        date_columns=("event_date", "actual_start", "planned_start"),
        rules=("R-CNT-001", "R-CNT-003"),
    ),
    LedgerTableSchema(
        name="downtime_events",
        title="Простои",
        required_columns=(
            "event_id",
            "source_file",
            "equipment_name",
            "downtime_category_raw",
            "is_reliability_downtime",
            "duration_hours",
        ),
        date_columns=("actual_start", "event_date"),
        rules=("R-CST-007", "R-CST-008"),
    ),
    LedgerTableSchema(
        name="cost_events",
        title="Затраты: бюджеты ПХД/ГБ и проводки УПА",
        required_columns=("event_id", "source_file", "cost_amount", "cost_kind"),
        date_columns=("event_date",),
        rules=("R-CST-002", "R-OUT-001"),
    ),
    LedgerTableSchema(
        name="mtr_writeoff_events",
        title="Списания МТР (MB25 и контур ЗАВ)",
        required_columns=(
            "event_id",
            "source_file",
            "material_id",
            "released_quantity",
            "unit_price",
            "order_id",
        ),
        date_columns=("event_date",),
        rules=("R-CST-005", "R-CST-002", "R-CST-003", "R-CST-004"),
    ),
    LedgerTableSchema(
        name="work_order_operation_events",
        title="Операции заказов SAP и плановая/фактическая трудоёмкость",
        required_columns=(
            "event_id",
            "source_file",
            "order_id",
            "operation_id",
            "planned_work_hours",
            "actual_work_hours",
        ),
        date_columns=("event_date", "planned_start"),
        rules=("R-CST-001", "R-CST-003"),
    ),
    LedgerTableSchema(
        name="material_requirements",
        title="Потребность в МТР из техкарт",
        required_columns=("event_id", "source_file", "material_id", "quantity", "unit"),
        date_columns=("event_date",),
        rules=("R-CST-004", "R-TKC-001"),
    ),
    LedgerTableSchema(
        name="tech_card_headers",
        title="Заголовки техкарт: точный ключ ЕО + счётчик, стратегия и нормативы",
        required_columns=(
            "event_id",
            "source_file",
            "equipment_id",
            "techcard_counter",
            "techcard_short_text",
            "strategy_id",
            "work_type_source",
        ),
        date_columns=("event_date",),
        rules=("R-TKC-001", "R-TKC-003", "R-TKC-007"),
    ),
    LedgerTableSchema(
        name="tech_card_operations",
        title="Операции техкарт",
        required_columns=("event_id", "source_file", "equipment_id", "operation_id", "work_hours"),
        date_columns=("event_date",),
        rules=("R-TKC-001", "R-TKC-003", "R-TKC-006", "R-TKC-008"),
    ),
    LedgerTableSchema(
        name="equipment_criticality_records",
        title="Критичность оборудования, «Код ABC»",
        required_columns=("criticality_record_id", "source_file", "equipment_id", "abc_code"),
        rules=("R-CST-003", "R-TKC-007"),
    ),
    LedgerTableSchema(
        name="equipment_registry",
        title="Реестр единиц оборудования и алиасов",
        required_columns=("equipment_id", "source_file", "equipment_name"),
        rules=("R-CST-003", "R-TKC-007"),
    ),
    LedgerTableSchema(
        name="fixed_asset_records",
        title="Реестр основных средств SAP",
        required_columns=("fixed_asset_record_id", "source_file", "fixed_asset_id"),
        rules=("R-CST-003",),
    ),
    LedgerTableSchema(
        name="sap_message_events",
        title="SAP-сообщения о дефектах",
        required_columns=("event_id", "source_file", "message_id", "message_type"),
        date_columns=("message_at", "created_at"),
        rules=("R-CNT-001", "R-CST-007"),
    ),
    LedgerTableSchema(
        name="equipment_condition_records",
        title="Состояние и износ оборудования",
        required_columns=("condition_record_id", "source_file", "equipment_id", "wear_pct"),
        rules=("R-CNT-004",),
    ),
    LedgerTableSchema(
        name="resource_capacity_records",
        title="Трудовые ресурсы и фонд времени",
        required_columns=("event_id", "source_file", "org_unit", "period"),
        rules=("R-OPT-003", "R-OPT-004"),
    ),
    LedgerTableSchema(
        name="source_documents",
        title="Каталог исходных файлов: тип, статус разбора, sha256",
        required_columns=("source_document_id", "relative_path", "sha256", "document_type", "parser_status"),
        rules=("R-OUT-002", "R-OUT-004"),
    ),
    LedgerTableSchema(
        name="source_records",
        title="Происхождение до строки и ячейки",
        required_columns=("event_id", "source_document_id", "record_kind"),
        rules=("R-OUT-002", "R-CNT-002"),
    ),
    LedgerTableSchema(
        name="parser_issues",
        title="Замечания парсера",
        required_columns=("issue_id", "source_file", "severity", "message"),
        rules=("R-OUT-001", "R-CNT-004"),
    ),
)

LEDGER_TABLES: dict[str, LedgerTableSchema] = {schema.name: schema for schema in _SCHEMAS}


def table_schema(name: str) -> LedgerTableSchema:
    """Схема таблицы ledger по имени; неизвестное имя — ошибка со списком известных."""

    try:
        return LEDGER_TABLES[name]
    except KeyError:
        known = ", ".join(sorted(LEDGER_TABLES))
        raise ValueError(f"неизвестная таблица ledger: {name!r}; известные: {known}") from None


__all__ = ["LEDGER_TABLES", "LedgerTableSchema", "table_schema"]
