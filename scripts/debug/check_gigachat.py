from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, load_config
from law_qa_rag.llm.base import LLMMessage
from law_qa_rag.llm.gigachat_client import GigaChatProvider, validate_model_available


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Проверить подключение к GigaChat",
        add_help=False,
    )
    parser._optionals.title = "параметры"
    parser._positionals.title = "позиционные аргументы"

    parser.add_argument("-h", "--help", action="help", help="Показать справку и выйти")
    parser.add_argument(
        "question",
        nargs="?",
        default="Проверка связи. Ответь одним коротким предложением.",
        help="Тестовый запрос к GigaChat",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="Путь к settings.yaml",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Показать доступные модели и выйти",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()
    config = load_config(args.settings)
    provider = GigaChatProvider(model=config.llm.model)

    if args.list_models:
        for model in provider.list_models():
            print(model)
        return

    validate_model_available(provider, config.llm.model)
    response = provider.complete(
        messages=[LLMMessage(role="user", content=args.question)],
        temperature=config.llm.temperature,
        max_tokens=min(config.llm.max_output_tokens, 300),
    )
    print(response.content)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
