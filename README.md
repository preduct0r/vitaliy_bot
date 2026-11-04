# Vitaliy Telegram Bot

Локальный Mac-бот для аудита и репетиции встречи. Это симулятор, не настоящий человек.

Установка: `uv sync --python 3.11 --extra dev`; на Mac нужны Poppler и LibreOffice для рендеринга.
Авторизация Codex остаётся на Mac. Токен и параметры хранятся вне Git в
`~/Library/Application Support/TOF Vitaly Bot/config.env`.

Для фактов TOF требуется настроенный внешний checkout `tof_v1` с SSH/Mutagen.
Укажите `VITALY_BOT_TOF_ROOT` — абсолютный путь к этому checkout (в окружении процесса).
Бот не содержит и не заменяет опубликованный ledger.

Запуск: `uv run python scripts/run_vitaly_bot.py`.
Проверка: `uv run pytest -q --tb=short`.
Полная инструкция: [docs/VITALY_TELEGRAM_BOT.md](docs/VITALY_TELEGRAM_BOT.md).

`SOURCE.json` фиксирует исходный commit TOF. Тег `baseline-before-point-corrections`
содержит версию до точечных исправлений. Откат кода не требует удаления новых
пользовательских данных; файлы состояния и резервные копии находятся вне Git.

Офлайн-пилот гипотез запускается только с подготовленными приватными индексом
корпуса и профилем S1. Перед генерацией он повторно проверяет фрагменты и хеши
источников, затем запрашивает покрытие текущего поколения через S2. Пример:

```
uv run --extra dev python scripts/hypotheses_pilot.py \
  --index "$WORKFLOW_INDEX" --profile "$WORKFLOW_PROFILE" \
  --knowledge-version "$WORKFLOW_KNOWLEDGE_VERSION" \
  --coverage-request "$WORKFLOW_COVERAGE_REQUEST" --scope "насосы" \
  --output "$WORKFLOW_PILOT_OUTPUT"
```

Запрос покрытия — JSON контракта S2, например
`{"version":1,"scope":{},"period":{}}`. Выходной каталог должен быть новым и
располагаться вне checkout: в нём сохраняются карточки, снимок покрытия и отчёт
проверки источников. Они могут содержать частные метаданные. Ошибка доступа к
серверу, неподтверждённые источники или отсутствие профиля блокируют пилот;
успех синтетических тестов не подтверждает проверку фактических источников.
