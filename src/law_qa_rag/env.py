from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


def load_project_dotenv(path: Path | None = None) -> bool:
    """Загружает локальный .env, не перезаписывая уже заданные env-переменные."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return False

    dotenv_path = path or DEFAULT_DOTENV_PATH
    if not dotenv_path.exists():
        return False
    return bool(load_dotenv(dotenv_path=dotenv_path, override=False))


def get_database_url(required: bool = True) -> str | None:
    """Возвращает DATABASE_URL или собирает его из POSTGRES_* переменных."""
    load_project_dotenv()

    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    db_name = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST") or "localhost"
    port = os.getenv("POSTGRES_PORT") or "5433"

    missing = [
        name
        for name, value in (
            ("POSTGRES_DB", db_name),
            ("POSTGRES_USER", user),
            ("POSTGRES_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        if not required:
            return None
        raise RuntimeError(
            "Нужны настройки подключения к БД: "
            + ", ".join(missing)
            + ". Задайте DATABASE_URL или заполните POSTGRES_* в .env."
        )

    return build_postgres_url(
        db_name=str(db_name),
        user=str(user),
        password=str(password),
        host=host,
        port=port,
    )


def build_postgres_url(
    db_name: str,
    user: str,
    password: str,
    host: str = "localhost",
    port: str = "5433",
) -> str:
    """Собирает PostgreSQL URL из отдельных параметров подключения."""
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(db_name, safe='')}"
    )
