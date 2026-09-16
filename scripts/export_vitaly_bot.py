#!/usr/bin/env python3
"""Export the explicit bot allowlist to a separate Git checkout (no user data)."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def export(root, destination):
    paths = ['src/tof_reliability/vitaly_bot', 'scripts/artifact_qa.py',
             'scripts/context_normalization.py', 'scripts/run_vitaly_bot.py',
             'scripts/install_vitaly_bot_launch_agent.py', 'scripts/evaluate_vitaly_profile_holdout.py',
             'docs/VITALY_TELEGRAM_BOT.md', 'discussions/vitaly',
             'tests/test_context_normalization.py', 'tests/test_artifact_qa.py']
    for optional in ['tests/test_vitaly_corrections.py', 'scripts/vitaly_snapshot.py',
                     'scripts/vitaly_release.py', 'scripts/export_vitaly_bot.py']:
        if (root / optional).exists():
            paths.append(optional)
    for name in paths:
        source, target = root / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copy2(source, target)
    for name in ['src/tof_reliability/__init__.py', 'scripts/__init__.py']:
        (destination / name).touch()
    (destination / '.python-version').write_text('3.11\n')
    (destination / 'pyproject.toml').write_text('''[project]
name = "vitaliy-bot"
version = "0.1.0"
requires-python = ">=3.10,<3.12"
dependencies = ["aiogram==3.30.0", "pillow>=11.3,<13"]
[project.optional-dependencies]
dev = ["pytest>=8.3.3,<9"]
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/tof_reliability", "scripts"]
[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
''')
    # Baseline uses the unchanged TOF client via an explicit external checkout.
    (destination / 'scripts/tof_remote_workspace.py').write_text('''import os
from pathlib import Path
import sys
root = Path(os.environ["VITALY_BOT_TOF_ROOT"]).expanduser().resolve()
os.chdir(root)
os.execv(str(root / ".venv/bin/python"), [str(root / ".venv/bin/python"), str(root / "scripts/tof_remote_workspace.py"), *sys.argv[1:]])
''')
    (destination / '.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n.pytest_cache/\n.env\nconfig.env\n*.sqlite*\ncases/\n*.log\ndist/\n')
    (destination / 'README.md').write_text('''# Vitaliy Telegram Bot

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
''')
    (destination / 'SOURCE.json').write_text(json.dumps({
        'tof_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_paths': paths}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    export(a.root, a.destination)
