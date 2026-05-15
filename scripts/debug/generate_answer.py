from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, load_config
from law_qa_rag.env import get_database_url
from law_qa_rag.generation import generate_answer, generated_answer_to_json
from law_qa_rag.llm.gigachat_client import (
    GigaChatProvider,
    ensure_gigachat_credentials,
    validate_model_available,
)


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Сгенерировать ответ по RAG-контексту",
        add_help=False,
    )
    parser._optionals.title = "параметры"
    parser._positionals.title = "позиционные аргументы"

    parser.add_argument("-h", "--help", action="help", help="Показать справку и выйти")
    parser.add_argument("question", type=str, help="Вопрос пользователя")
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="Путь к settings.yaml",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=get_database_url(required=False),
        help="PostgreSQL URL. Если не передан, берется из DATABASE_URL или POSTGRES_* в .env.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Устройство для embedding-модели",
    )
    parser.add_argument("--act", type=str, default=None, help="Фильтр по названию акта")
    parser.add_argument("--article-no", type=str, default=None, help="Фильтр по номеру статьи")
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()
    if not args.db_url:
        raise ValueError(
            "Нужен URL БД. Передайте --db-url или заполните DATABASE_URL/POSTGRES_* в .env."
        )

    config = load_config(args.settings)
    provider = GigaChatProvider(model=config.llm.model)
    ensure_gigachat_credentials()
    validate_model_available(provider, config.llm.model)

    result = generate_answer(
        question=args.question,
        db_url=args.db_url,
        config=config,
        provider=provider,
        device=args.device,
        act_filter=args.act,
        article_no=args.article_no,
    )
    print(generated_answer_to_json(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
