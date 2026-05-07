from __future__ import annotations

import argparse
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


SPARSE_SEARCH_SQL = """
WITH q AS (
    SELECT websearch_to_tsquery('russian', %(question)s::text) AS query
)
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
    ts_rank_cd(c.search_vector, q.query) AS rank,
    ts_headline(
        'russian',
        c.text,
        q.query,
        %(headline_options)s::text
    ) AS snippet,
    c.text AS full_text
FROM chunks c
JOIN acts a ON a.id = c.act_id
CROSS JOIN q
WHERE c.search_vector @@ q.query
  AND (%(act_filter)s::text IS NULL OR a.title ILIKE '%%' || %(act_filter)s::text || '%%')
  AND (%(article_no)s::text IS NULL OR c.article_no = %(article_no)s::text)
ORDER BY rank DESC, c.id
LIMIT %(limit)s;
"""


def positive_int(value: str) -> int:
    """Проверяет положительные целые CLI-аргументы."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("значение должно быть целым числом") from exc

    if result <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше 0")

    return result


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы."""
    parser = argparse.ArgumentParser(
        description="Отладить PostgreSQL sparse search по юридическим chunks",
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
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL URL. Можно также передать через DATABASE_URL.",
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
        "--snippet-words",
        type=positive_int,
        default=80,
        help="Примерный максимум слов в подсвеченном snippet",
    )

    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Показать полный текст chunk вместо короткого snippet",
    )

    return parser.parse_args()


def build_headline_options(snippet_words: int) -> str:
    """Собирает параметры PostgreSQL ts_headline."""
    max_words = max(20, snippet_words)
    min_words = min(25, max_words)

    return (
        "StartSel=<<, "
        "StopSel=>>, "
        f"MaxWords={max_words}, "
        f"MinWords={min_words}, "
        "ShortWord=3, "
        "HighlightAll=false"
    )


def search_sparse(
    db_url: str,
    question: str,
    limit: int,
    act_filter: str | None,
    article_no: str | None,
    snippet_words: int,
) -> list[dict[str, Any]]:
    """Выполняет полнотекстовый sparse search в PostgreSQL."""
    params = {
        "question": question,
        "limit": limit,
        "act_filter": act_filter,
        "article_no": article_no,
        "headline_options": build_headline_options(snippet_words),
    }

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(SPARSE_SEARCH_SQL, params)
            return list(cur.fetchall())


def print_result(row: dict[str, Any], rank_no: int, show_full_text: bool) -> None:
    """Печатает один найденный chunk."""
    print("=" * 100)
    print(f"#{rank_no}")
    print(f"chunk_id:     {row['chunk_id']}")
    print(f"act:          {row['act_title']} от {row['doc_date']} № {row['doc_number']}")
    print(f"chunk_index:  {row['chunk_index']}")
    print(f"article_no:   {row['article_no']}")
    print(f"clause_range: {row['clause_range']}")
    print(f"token_count:  {row['token_count']}")
    print(f"rank:         {float(row['rank']):.6f}")
    print(f"structure:    {row['structure_ref']}")
    print()

    if show_full_text:
        print(row["full_text"])
    else:
        print(row["snippet"])


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()

    question = args.question.strip()
    if not question:
        raise ValueError("Запрос не должен быть пустым")

    if not args.db_url:
        raise ValueError(
            "Нужен URL БД. Передайте --db-url или задайте DATABASE_URL."
        )

    rows = search_sparse(
        db_url=args.db_url,
        question=question,
        limit=args.limit,
        act_filter=args.act,
        article_no=args.article_no,
        snippet_words=args.snippet_words,
    )

    print(f"[OK] запрос: {question}")
    if args.act:
        print(f"[OK] фильтр по акту: {args.act}")
    if args.article_no:
        print(f"[OK] фильтр по статье: {args.article_no}")
    print(f"[OK] результатов: {len(rows)}")
    print()

    if not rows:
        print("Sparse search ничего не нашел.")
        print("Попробуйте меньше слов, другую формулировку или уберите фильтры.")
        return

    for i, row in enumerate(rows, start=1):
        print_result(row, i, args.show_full_text)


if __name__ == "__main__":
    main()
