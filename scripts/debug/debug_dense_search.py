from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import torch
import yaml
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

from law_qa_rag.env import get_database_url


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.yaml"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024


DENSE_SEARCH_SQL = """
SELECT
    c.id AS chunk_id,
    a.id AS act_id,
    a.title AS act_title,
    a.doc_number,
    a.doc_date,
    c.chunk_index,
    c.structure_ref,
    c.article_no,
    c.clause_range,
    c.token_count,
    c.embedding_model,
    c.embedding <=> %(query_embedding)s::vector AS distance,
    1 - (c.embedding <=> %(query_embedding)s::vector) AS score,
    c.text AS full_text
FROM chunks c
JOIN acts a ON a.id = c.act_id
WHERE c.embedding IS NOT NULL
  AND c.embedding_model = %(embedding_model)s::text
  AND (%(act_filter)s::text IS NULL OR a.title ILIKE '%%' || %(act_filter)s::text || '%%')
  AND (%(article_no)s::text IS NULL OR c.article_no = %(article_no)s::text)
ORDER BY c.embedding <=> %(query_embedding)s::vector ASC, c.id
LIMIT %(limit)s;
"""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Настройки модели embeddings, полученные из YAML и CLI."""

    model_name: str = DEFAULT_MODEL_NAME
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    settings_path: str | None = None


def coerce_positive_int(value: Any, field_name: str) -> int:
    """Преобразует значение в положительное целое число."""
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} должно быть целым числом") from exc

    if result <= 0:
        raise ValueError(f"{field_name} должно быть больше 0")

    return result


def positive_int(value: str) -> int:
    """Проверяет положительные целые CLI-аргументы."""
    try:
        return coerce_positive_int(value, "значение")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def read_settings(path: Path | None) -> dict[str, Any]:
    """Читает YAML-настройки проекта, если файл существует."""
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Файл настроек должен быть YAML mapping: {path}")
    return data


def build_embedding_config(
    settings: dict[str, Any],
    settings_path: Path | None,
) -> EmbeddingConfig:
    """Собирает настройки embeddings из settings.yaml."""
    embedding = settings.get("embedding") or {}
    if not isinstance(embedding, dict):
        embedding = {}

    model_name = (
        embedding.get("embedding_model")
        or embedding.get("model")
        or DEFAULT_MODEL_NAME
    )
    embedding_dim = (
        embedding.get("embedding_dim")
        or embedding.get("dim")
        or DEFAULT_EMBEDDING_DIM
    )

    return EmbeddingConfig(
        model_name=str(model_name),
        embedding_dim=coerce_positive_int(embedding_dim, "embedding_dim"),
        settings_path=str(settings_path) if settings_path and settings_path.exists() else None,
    )


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Отладить pgvector dense search по юридическим chunks",
        add_help=False,
    )
    parser._optionals.title = "параметры"
    parser._positionals.title = "позиционные аргументы"

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Показать справку и выйти",
    )

    parser.add_argument(
        "question",
        type=str,
        help="Поисковый запрос, например: 'водные объекты общего пользования'",
    )

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
        "--model",
        type=str,
        default=None,
        help="Переопределить SentenceTransformer-модель из settings.yaml",
    )

    parser.add_argument(
        "--embedding-dim",
        type=positive_int,
        default=None,
        help="Переопределить размерность embeddings из settings.yaml",
    )

    parser.add_argument(
        "--limit",
        type=positive_int,
        default=10,
        help="Сколько результатов показать",
    )

    parser.add_argument(
        "--act",
        type=str,
        default=None,
        help="Фильтр по названию акта, например: 'Водный кодекс'",
    )

    parser.add_argument(
        "--article-no",
        type=str,
        default=None,
        help="Фильтр по точному номеру статьи, например: '6'",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Устройство для модели embeddings",
    )

    parser.add_argument(
        "--preview-chars",
        type=positive_int,
        default=900,
        help="Сколько символов текста chunk показывать в коротком фрагменте",
    )

    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Показать полный текст chunk вместо короткого фрагмента",
    )

    return parser.parse_args()


def detect_device(requested: str) -> str:
    """Выбирает устройство для запуска модели."""
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(model_name: str, device: str) -> SentenceTransformer:
    """Загружает SentenceTransformer-модель."""
    print(f"[OK] модель: {model_name}")
    print(f"[OK] устройство: {device}")
    return SentenceTransformer(model_name, device=device)


def embedding_to_pgvector(value: np.ndarray) -> str:
    """Преобразует numpy-вектор в текстовый формат pgvector."""
    return "[" + ",".join(f"{float(x):.8f}" for x in value) + "]"


def encode_query(
    model: SentenceTransformer,
    question: str,
    expected_dim: int,
) -> np.ndarray:
    """Кодирует поисковый запрос и проверяет размерность embedding."""
    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if embedding.ndim != 2:
        raise RuntimeError(
            f"Ожидался 2D массив embeddings, получена shape {embedding.shape}"
        )

    if embedding.shape[0] != 1:
        raise RuntimeError(
            f"Ожидался один query embedding, получена shape {embedding.shape}"
        )

    if embedding.shape[1] != expected_dim:
        raise RuntimeError(
            f"Ожидалась размерность embeddings {expected_dim}, получено {embedding.shape[1]}"
        )

    return embedding[0]


def count_embeddings(cur: psycopg.Cursor) -> tuple[int, int, int]:
    """Считает chunks с embeddings и без них."""
    cur.execute("""
        SELECT
            count(*) AS total,
            count(*) FILTER (WHERE embedding IS NOT NULL) AS with_embeddings,
            count(*) FILTER (WHERE embedding IS NULL) AS without_embeddings
        FROM chunks;
    """)

    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Не удалось получить статистику embeddings")

    return (
        int(row["total"]),
        int(row["with_embeddings"]),
        int(row["without_embeddings"]),
    )


def count_embeddings_for_model(cur: psycopg.Cursor, model_name: str) -> int:
    """Считает chunks с embeddings выбранной модели."""
    cur.execute(
        """
        SELECT count(*) AS with_model_embeddings
        FROM chunks
        WHERE embedding IS NOT NULL
          AND embedding_model = %s;
        """,
        (model_name,),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Не удалось получить статистику embeddings модели")
    return int(row["with_model_embeddings"])


def search_dense(
    db_url: str,
    query_embedding: np.ndarray,
    model_name: str,
    limit: int,
    act_filter: str | None,
    article_no: str | None,
) -> list[dict[str, Any]]:
    """Выполняет dense search по pgvector embeddings выбранной модели."""
    params = {
        "query_embedding": embedding_to_pgvector(query_embedding),
        "embedding_model": model_name,
        "limit": limit,
        "act_filter": act_filter,
        "article_no": article_no,
    }

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            total, with_embeddings, without_embeddings = count_embeddings(cur)
            with_model_embeddings = count_embeddings_for_model(cur, model_name)
            print(f"[OK] chunks всего: {total}")
            print(f"[OK] chunks с embeddings: {with_embeddings}")
            print(f"[OK] chunks без embeddings: {without_embeddings}")
            print(f"[OK] embeddings модели {model_name}: {with_model_embeddings}")

            if with_embeddings == 0:
                raise RuntimeError(
                    "В chunks нет embeddings. Сначала запустите scripts/pipeline/embed_chunks.py."
                )
            if with_model_embeddings == 0:
                raise RuntimeError(
                    f"Нет embeddings для модели {model_name}. "
                    "Запустите scripts/pipeline/embed_chunks.py с этой моделью."
                )

            cur.execute(DENSE_SEARCH_SQL, params)
            return list(cur.fetchall())


def make_preview(text: str, preview_chars: int) -> str:
    """Обрезает текст chunk до preview-длины."""
    text = text.strip()
    if len(text) <= preview_chars:
        return text

    return text[:preview_chars].rstrip() + "\n..."


def print_result(
    row: dict[str, Any],
    rank_no: int,
    show_full_text: bool,
    preview_chars: int,
) -> None:
    """Печатает один найденный chunk."""
    print("=" * 100)
    print(f"#{rank_no}")
    print(f"chunk_id:        {row['chunk_id']}")
    print(f"act:             {row['act_title']} от {row['doc_date']} № {row['doc_number']}")
    print(f"chunk_index:     {row['chunk_index']}")
    print(f"article_no:      {row['article_no']}")
    print(f"clause_range:    {row['clause_range']}")
    print(f"token_count:     {row['token_count']}")
    print(f"embedding_model: {row['embedding_model']}")
    print(f"distance:        {float(row['distance']):.6f}")
    print(f"score:           {float(row['score']):.6f}")
    print(f"structure:       {row['structure_ref']}")
    print()

    if show_full_text:
        print(row["full_text"])
    else:
        print(make_preview(row["full_text"], preview_chars))


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()

    question = args.question.strip()
    if not question:
        raise ValueError("Запрос не должен быть пустым")

    if not args.db_url:
        raise ValueError(
            "Нужен URL БД. Передайте --db-url или заполните DATABASE_URL/POSTGRES_* в .env."
        )

    settings = read_settings(args.settings)
    config = build_embedding_config(settings, args.settings)
    if args.model is not None:
        config = replace(config, model_name=args.model)
    if args.embedding_dim is not None:
        config = replace(config, embedding_dim=args.embedding_dim)

    print(f"[OK] settings.yaml: {config.settings_path or 'не используется'}")
    print(f"[OK] размерность embeddings: {config.embedding_dim}")

    device = detect_device(args.device)
    model = load_model(config.model_name, device)
    query_embedding = encode_query(model, question, config.embedding_dim)

    rows = search_dense(
        db_url=args.db_url,
        query_embedding=query_embedding,
        model_name=config.model_name,
        limit=args.limit,
        act_filter=args.act,
        article_no=args.article_no,
    )

    print(f"[OK] запрос: {question}")
    if args.act:
        print(f"[OK] фильтр по акту: {args.act}")
    if args.article_no:
        print(f"[OK] фильтр по статье: {args.article_no}")
    print(f"[OK] результатов: {len(rows)}")
    print()

    if not rows:
        print("Dense search ничего не нашел.")
        print("Попробуйте убрать фильтры --act/--article-no.")
        return

    for i, row in enumerate(rows, start=1):
        print_result(
            row=row,
            rank_no=i,
            show_full_text=args.show_full_text,
            preview_chars=args.preview_chars,
        )


if __name__ == "__main__":
    main()
