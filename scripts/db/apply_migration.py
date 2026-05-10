from __future__ import annotations

import argparse
from pathlib import Path

import psycopg

from law_qa_rag.env import get_database_url


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Применить один SQL migration-файл к PostgreSQL",
    )
    parser.add_argument("migration_file", type=Path, help="Путь к .sql migration-файлу")
    parser.add_argument(
        "--db-url",
        type=str,
        default=get_database_url(required=False),
        help="PostgreSQL URL. Если не передан, берется из DATABASE_URL или POSTGRES_* в .env.",
    )
    return parser.parse_args()


def read_migration_sql(path: Path) -> str:
    """Читает SQL migration-файл."""
    if not path.exists():
        raise FileNotFoundError(f"Migration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Migration path is not a file: {path}")
    return path.read_text(encoding="utf-8")


def apply_migration(db_url: str, sql_text: str) -> None:
    """Выполняет SQL migration в PostgreSQL."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()
    if not args.db_url:
        raise ValueError("Нужен URL БД. Передайте --db-url или заполните DATABASE_URL/POSTGRES_* в .env.")

    sql_text = read_migration_sql(args.migration_file)
    apply_migration(args.db_url, sql_text)
    print(f"[OK] migration applied: {args.migration_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
