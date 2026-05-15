from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, load_config
from law_qa_rag.env import get_database_url
from law_qa_rag.generation import apply_token_budget
from law_qa_rag.llm.gigachat_client import (
    GigaChatProvider,
    ensure_gigachat_credentials,
    validate_model_available,
)
from law_qa_rag.retrieval import retrieve_chunks


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Посчитать токены prompt перед генерацией ответа",
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

    chunks = retrieve_chunks(
        db_url=args.db_url,
        question=args.question,
        retrieval_config=config.retrieval,
        embedding_config=config.embedding,
        device=args.device,
        act_filter=args.act,
        article_no=args.article_no,
    )
    budget = apply_token_budget(args.question, chunks, config, provider)
    payload = {
        "question": args.question,
        "retrieval_method": config.retrieval.method,
        "context_token_budget": config.llm.context_token_budget,
        "total_tokens": budget.total_tokens,
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "selected_chunk_ids": [chunk.chunk_id for chunk in budget.selected_chunks],
        "dropped_chunk_ids": budget.dropped_chunk_ids,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
