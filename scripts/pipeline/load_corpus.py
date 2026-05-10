from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from law_qa_rag.env import get_database_url


INSERT_ACT_SQL = """
INSERT INTO acts (
    canonical_key,
    act_kind,
    doc_type,
    title,
    doc_number,
    doc_date,
    official_text_kind,
    edition_as_of,
    edition_note,
    status,
    has_future_editions,
    source_file,
    source_system
)
VALUES (
    %(canonical_key)s,
    %(act_kind)s,
    %(doc_type)s,
    %(title)s,
    %(doc_number)s,
    %(doc_date)s,
    %(official_text_kind)s,
    %(edition_as_of)s,
    %(edition_note)s,
    %(status)s,
    %(has_future_editions)s,
    %(source_file)s,
    %(source_system)s
)
ON CONFLICT (canonical_key) DO UPDATE SET
    act_kind = EXCLUDED.act_kind,
    doc_type = EXCLUDED.doc_type,
    title = EXCLUDED.title,
    doc_number = EXCLUDED.doc_number,
    doc_date = EXCLUDED.doc_date,
    official_text_kind = EXCLUDED.official_text_kind,
    edition_as_of = EXCLUDED.edition_as_of,
    edition_note = EXCLUDED.edition_note,
    status = EXCLUDED.status,
    has_future_editions = EXCLUDED.has_future_editions,
    source_file = EXCLUDED.source_file,
    source_system = EXCLUDED.source_system;
"""


INSERT_CHUNK_SQL = """
INSERT INTO chunks (
    act_id,
    chunk_index,
    text,
    structure_ref,
    article_no,
    clause_range,
    source_anchors,
    start_node_order,
    end_node_order,
    token_count,
    hash
)
VALUES (
    %(act_id)s,
    %(chunk_index)s,
    %(text)s,
    %(structure_ref)s,
    %(article_no)s,
    %(clause_range)s,
    %(source_anchors)s,
    %(start_node_order)s,
    %(end_node_order)s,
    %(token_count)s,
    %(hash)s
)
ON CONFLICT (act_id, chunk_index) DO UPDATE SET
    text = EXCLUDED.text,
    structure_ref = EXCLUDED.structure_ref,
    article_no = EXCLUDED.article_no,
    clause_range = EXCLUDED.clause_range,
    source_anchors = EXCLUDED.source_anchors,
    start_node_order = EXCLUDED.start_node_order,
    end_node_order = EXCLUDED.end_node_order,
    token_count = EXCLUDED.token_count,
    hash = EXCLUDED.hash,
    embedding = NULL,
    embedding_model = NULL;
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")

            rows.append(obj)

    return rows


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")

    return result


def normalize_act_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_key": row.get("canonical_key"),
        "act_kind": row.get("act_kind"),
        "doc_type": row.get("doc_type"),
        "title": row.get("title"),
        "doc_number": row.get("doc_number"),
        "doc_date": row.get("doc_date"),
        "official_text_kind": row.get("official_text_kind"),
        "edition_as_of": row.get("edition_as_of"),
        "edition_note": row.get("edition_note"),
        "status": row.get("status") or "unknown",
        "has_future_editions": bool(row.get("has_future_editions")),
        "source_file": row.get("source_file"),
        "source_system": row.get("source_system") or "pravo.gov.ru html export",
    }


def validate_acts(acts: list[dict[str, Any]]) -> None:
    required_fields = [
        "canonical_key",
        "act_kind",
        "doc_type",
        "title",
        "doc_number",
        "doc_date",
        "edition_as_of",
        "source_file",
    ]

    seen_keys: set[str] = set()

    for i, act in enumerate(acts, start=1):
        for field in required_fields:
            if not act.get(field):
                raise ValueError(f"acts.jsonl row {i}: missing required field {field}")

        key = act["canonical_key"]

        if key in seen_keys:
            raise ValueError(f"acts.jsonl row {i}: duplicate canonical_key {key}")

        seen_keys.add(key)


def validate_chunks(
    chunks: list[dict[str, Any]],
    act_keys: set[str],
) -> None:
    seen_pairs: set[tuple[str, int]] = set()
    seen_hashes: set[str] = set()

    for i, chunk in enumerate(chunks, start=1):
        canonical_key = chunk.get("canonical_key")
        chunk_index = chunk.get("chunk_index")
        text = chunk.get("text")
        token_count = chunk.get("token_count")
        hash_value = chunk.get("hash")

        if not canonical_key:
            raise ValueError(f"chunks.jsonl row {i}: missing canonical_key")

        if canonical_key not in act_keys:
            raise ValueError(
                f"chunks.jsonl row {i}: canonical_key not found in acts.jsonl: {canonical_key}"
            )

        if chunk_index is None:
            raise ValueError(f"chunks.jsonl row {i}: missing chunk_index")

        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError(f"chunks.jsonl row {i}: invalid chunk_index {chunk_index}")

        if not text or not str(text).strip():
            raise ValueError(f"chunks.jsonl row {i}: empty text")

        if not isinstance(token_count, int) or token_count <= 0:
            raise ValueError(f"chunks.jsonl row {i}: invalid token_count {token_count}")

        if not hash_value:
            raise ValueError(f"chunks.jsonl row {i}: missing hash")

        pair = (canonical_key, chunk_index)
        if pair in seen_pairs:
            raise ValueError(f"chunks.jsonl row {i}: duplicate chunk key {pair}")

        seen_pairs.add(pair)

        if hash_value in seen_hashes:
            raise ValueError(f"chunks.jsonl row {i}: duplicate hash {hash_value}")

        seen_hashes.add(hash_value)

        source_anchors = chunk.get("source_anchors") or []
        if not isinstance(source_anchors, list):
            raise ValueError(f"chunks.jsonl row {i}: source_anchors must be list")

        start_order = chunk.get("start_node_order")
        end_order = chunk.get("end_node_order")

        if (
            start_order is not None
            and end_order is not None
            and end_order < start_order
        ):
            raise ValueError(
                f"chunks.jsonl row {i}: end_node_order < start_node_order"
            )


def normalize_chunk_row(
    chunk: dict[str, Any],
    act_id_by_key: dict[str, int],
) -> dict[str, Any]:
    canonical_key = chunk.get("canonical_key")

    if canonical_key not in act_id_by_key:
        raise ValueError(f"Unknown canonical_key in chunk: {canonical_key}")

    source_anchors = chunk.get("source_anchors") or []

    if not isinstance(source_anchors, list):
        raise ValueError(
            f"source_anchors must be list for canonical_key={canonical_key}, "
            f"chunk_index={chunk.get('chunk_index')}"
        )

    return {
        "act_id": act_id_by_key[canonical_key],
        "chunk_index": chunk.get("chunk_index"),
        "text": chunk.get("text"),
        "structure_ref": chunk.get("structure_ref"),
        "article_no": chunk.get("article_no"),
        "clause_range": chunk.get("clause_range"),
        "source_anchors": Jsonb(source_anchors),
        "start_node_order": chunk.get("start_node_order"),
        "end_node_order": chunk.get("end_node_order"),
        "token_count": chunk.get("token_count"),
        "hash": chunk.get("hash"),
    }


def reset_corpus(cur: psycopg.Cursor) -> None:
    cur.execute("TRUNCATE TABLE acts, chunks RESTART IDENTITY CASCADE;")


def delete_chunks_for_acts(cur: psycopg.Cursor, act_ids: set[int]) -> int:
    if not act_ids:
        return 0

    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in act_ids)
    cur.execute(
        sql.SQL("DELETE FROM chunks WHERE act_id IN ({})").format(placeholders),
        sorted(act_ids),
    )
    return cur.rowcount


def load_acts(cur: psycopg.Cursor, acts: list[dict[str, Any]]) -> None:
    rows = [normalize_act_row(act) for act in acts]
    cur.executemany(INSERT_ACT_SQL, rows)


def get_act_id_map(cur: psycopg.Cursor) -> dict[str, int]:
    cur.execute("SELECT id, canonical_key FROM acts;")
    rows = cur.fetchall()

    return {
        canonical_key: act_id
        for act_id, canonical_key in rows
    }


def load_chunks(
    cur: psycopg.Cursor,
    chunks: list[dict[str, Any]],
    act_id_by_key: dict[str, int],
    batch_size: int,
) -> None:
    batch: list[dict[str, Any]] = []
    loaded = 0

    for chunk in chunks:
        row = normalize_chunk_row(chunk, act_id_by_key)
        batch.append(row)

        if len(batch) >= batch_size:
            cur.executemany(INSERT_CHUNK_SQL, batch)
            loaded += len(batch)
            print(f"[OK] loaded chunks: {loaded}")
            batch.clear()

    if batch:
        cur.executemany(INSERT_CHUNK_SQL, batch)
        loaded += len(batch)
        print(f"[OK] loaded chunks: {loaded}")


def print_db_stats(cur: psycopg.Cursor) -> None:
    cur.execute("SELECT count(*) FROM acts;")
    acts_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM chunks;")
    chunks_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL;")
    chunks_without_embeddings = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM chunks WHERE length(trim(text)) = 0;")
    empty_chunks = cur.fetchone()[0]

    print(f"[OK] database acts: {acts_count}")
    print(f"[OK] database chunks: {chunks_count}")
    print(f"[OK] chunks without embeddings: {chunks_without_embeddings}")
    print(f"[OK] empty chunks: {empty_chunks}")

    cur.execute("""
        SELECT a.title, count(*) AS chunk_count
        FROM chunks c
        JOIN acts a ON a.id = c.act_id
        GROUP BY a.title
        ORDER BY chunk_count DESC
        LIMIT 10;
    """)

    print("[OK] top acts by chunk count:")
    for title, count in cur.fetchall():
        print(f"  {count:>6}  {title}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load prepared legal corpus JSONL files into PostgreSQL"
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory with acts.jsonl and chunks.jsonl",
    )

    parser.add_argument(
        "--db-url",
        type=str,
        default=get_database_url(required=False),
        help="PostgreSQL URL. Если не передан, берется из DATABASE_URL или POSTGRES_* в .env.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate acts/chunks before loading",
    )

    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1000,
        help="Batch size for chunk inserts",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only read and validate files, do not write to database",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    acts_path = input_dir / "acts.jsonl"
    chunks_path = input_dir / "chunks.jsonl"

    if not acts_path.exists():
        raise FileNotFoundError(f"Missing file: {acts_path}")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing file: {chunks_path}")

    print(f"[OK] reading {acts_path}")
    acts = read_jsonl(acts_path)

    print(f"[OK] reading {chunks_path}")
    chunks = read_jsonl(chunks_path)

    print(f"[OK] read acts: {len(acts)}")
    print(f"[OK] read chunks: {len(chunks)}")

    validate_acts(acts)

    act_keys = {act["canonical_key"] for act in acts}
    validate_chunks(chunks, act_keys)

    print("[OK] validation passed")

    if args.dry_run:
        print("[OK] dry run finished, nothing was written to database")
        return

    if not args.db_url:
        raise ValueError(
            "Нужен URL БД. Передайте --db-url или заполните DATABASE_URL/POSTGRES_* в .env."
        )

    with psycopg.connect(args.db_url) as conn:
        with conn.cursor() as cur:
            if args.reset:
                reset_corpus(cur)
                print("[OK] reset corpus tables")

            load_acts(cur, acts)
            print(f"[OK] loaded acts: {len(acts)}")

            act_id_by_key = get_act_id_map(cur)

            missing_keys = act_keys - set(act_id_by_key.keys())
            if missing_keys:
                raise RuntimeError(f"Some acts were not loaded: {sorted(missing_keys)[:5]}")

            loaded_act_ids = {act_id_by_key[key] for key in act_keys}
            deleted_chunks = delete_chunks_for_acts(cur, loaded_act_ids)
            print(f"[OK] deleted existing chunks for loaded acts: {deleted_chunks}")

            load_chunks(
                cur=cur,
                chunks=chunks,
                act_id_by_key=act_id_by_key,
                batch_size=args.batch_size,
            )

            print_db_stats(cur)

    print("[OK] corpus loading finished")


if __name__ == "__main__":
    main()
