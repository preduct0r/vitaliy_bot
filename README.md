# Vitaliy Telegram Bot

Локальный Mac-бот для аудита и репетиции встречи. Это симулятор, не настоящий человек.

Установка: `uv sync --extra dev`; на Mac нужны Poppler и LibreOffice для рендеринга.
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
