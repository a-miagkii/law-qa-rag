from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from law_qa_rag.env import get_database_url


DEFAULT_INPUT = Path("eval/eval_questions.jsonl")
DEFAULT_OUTPUT = Path("eval/gold_resolved.jsonl")
DEFAULT_UNRESOLVED = Path("eval/unresolved_gold.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разрешить предварительную разметку act/article в реальные chunks.id",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unresolved", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument(
        "--db-url",
        type=str,
        default=get_database_url(required=False),
        help="PostgreSQL URL. Если не указан, берется из DATABASE_URL или POSTGRES_* в .env.",
    )
    parser.add_argument(
        "--no-title-fallback",
        action="store_true",
        help="Не искать по названию акта, если canonical_key не дал результатов.",
    )
    parser.add_argument(
        "--max-matches-per-ref",
        type=int,
        default=20,
        help="Порог предупреждения о слишком широком совпадении по одной ссылке.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSON в {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Ожидался JSON-объект в {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def normalize_refs(item: dict[str, Any]) -> list[dict[str, Any]]:
    refs = item.get("gold_refs")
    if isinstance(refs, list) and refs:
        normalized = []
        for ref in refs:
            if isinstance(ref, dict):
                normalized.append(ref)
        return normalized

    expected_act_title = item.get("expected_act_title")
    expected_canonical_key = item.get("expected_canonical_key")
    expected_article_no = item.get("expected_article_no")
    if expected_act_title or expected_canonical_key or expected_article_no:
        return [
            {
                "act_title": expected_act_title,
                "canonical_key": expected_canonical_key,
                "article_no": expected_article_no,
            }
        ]
    return []


def find_chunks_for_ref(
    cur: psycopg.Cursor,
    ref: dict[str, Any],
    use_canonical_key: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    canonical_key = clean_str(ref.get("canonical_key"))
    act_title = clean_str(ref.get("act_title"))
    article_no = clean_str(ref.get("article_no"))
    text_any = ref.get("expected_text_contains_any") or ref.get("text_contains_any")

    where: list[str] = []
    params: dict[str, Any] = {}
    resolved_by = ""

    if use_canonical_key and canonical_key:
        where.append("a.canonical_key = %(canonical_key)s")
        params["canonical_key"] = canonical_key
        resolved_by = "canonical_key"
    elif act_title:
        where.append("a.title ILIKE %(act_title)s")
        params["act_title"] = f"%{act_title}%"
        resolved_by = "act_title"
    else:
        return [], "missing_act_identifier"

    if article_no:
        where.append("c.article_no = %(article_no)s")
        params["article_no"] = article_no

    query = f"""
        SELECT
            c.id AS chunk_id,
            a.id AS act_id,
            a.canonical_key,
            a.title AS act_title,
            a.doc_number,
            a.doc_date,
            c.chunk_index,
            c.article_no,
            c.structure_ref,
            c.clause_range,
            c.token_count,
            c.text
        FROM chunks c
        JOIN acts a ON a.id = c.act_id
        WHERE {" AND ".join(where)}
        ORDER BY a.canonical_key, c.chunk_index, c.id;
    """
    cur.execute(query, params)
    rows = [dict(row) for row in cur.fetchall()]

    if text_any:
        if isinstance(text_any, str):
            needles = [text_any]
        elif isinstance(text_any, list):
            needles = [str(x) for x in text_any if str(x).strip()]
        else:
            needles = []
        if needles:
            lowered = [needle.lower() for needle in needles]
            filtered = [
                row
                for row in rows
                if any(needle in str(row.get("text") or "").lower() for needle in lowered)
            ]
            if filtered:
                rows = filtered
            else:
                return [], f"{resolved_by}_text_filter_no_match"

    for row in rows:
        # Полный текст не нужен в эталонном файле, оставляем короткий фрагмент для ручной проверки.
        text = str(row.pop("text", "") or "")
        row["text_preview"] = text[:500].replace("\n", " ")

    return rows, resolved_by


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_items(
    db_url: str,
    items: list[dict[str, Any]],
    allow_title_fallback: bool,
    max_matches_per_ref: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_items: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for item in items:
                item_id = str(item.get("id") or "")
                refs = normalize_refs(item)
                is_answerable = bool(item.get("is_answerable", True))
                all_chunk_ids: list[int] = []
                resolved_refs: list[dict[str, Any]] = []
                warnings: list[str] = []
                failed_refs = 0

                if not refs:
                    status = "skipped_unanswerable" if not is_answerable else "unresolved"
                    if is_answerable:
                        unresolved_rows.append(make_unresolved_row(item, None, "no_gold_refs"))
                    out = dict(item)
                    out["gold_chunk_ids"] = list(
                        dict.fromkeys(
                            int(x) for x in item.get("gold_chunk_ids", []) if str(x).isdigit()
                        )
                    )
                    out["resolved_refs"] = []
                    out["resolution_status"] = status
                    out["resolution_warnings"] = warnings
                    resolved_items.append(out)
                    continue

                for idx, ref in enumerate(refs, start=1):
                    rows, resolved_by = find_chunks_for_ref(cur, ref, use_canonical_key=True)
                    if not rows and allow_title_fallback:
                        rows, resolved_by = find_chunks_for_ref(cur, ref, use_canonical_key=False)
                        if rows:
                            warnings.append(f"{item_id}: ref {idx} resolved by title fallback")

                    ref_out = dict(ref)
                    ref_out["resolved_by"] = resolved_by
                    ref_out["match_count"] = len(rows)
                    ref_out["resolved_chunk_ids"] = [int(row["chunk_id"]) for row in rows]
                    ref_out["matches_preview"] = rows[:5]
                    resolved_refs.append(ref_out)

                    if not rows:
                        failed_refs += 1
                        unresolved_rows.append(
                            make_unresolved_row(item, ref, f"no_chunks_found:{resolved_by}", idx)
                        )
                    else:
                        if len(rows) > max_matches_per_ref:
                            warnings.append(
                                f"{item_id}: ref {idx} returned {len(rows)} chunks; check article split manually"
                            )
                        all_chunk_ids.extend(int(row["chunk_id"]) for row in rows)

                existing_ids = []
                for raw_id in item.get("gold_chunk_ids", []) or []:
                    try:
                        existing_ids.append(int(raw_id))
                    except (TypeError, ValueError):
                        warnings.append(
                            f"{item_id}: non-integer existing gold_chunk_id ignored: {raw_id!r}"
                        )

                all_chunk_ids = list(dict.fromkeys(existing_ids + all_chunk_ids))

                if failed_refs == 0 and all_chunk_ids:
                    status = "resolved"
                elif failed_refs < len(refs) and all_chunk_ids:
                    status = "partial"
                else:
                    status = "unresolved"

                out = dict(item)
                out["gold_chunk_ids"] = all_chunk_ids
                out["resolved_refs"] = resolved_refs
                out["resolution_status"] = status
                out["resolution_warnings"] = warnings
                resolved_items.append(out)

    return resolved_items, unresolved_rows


def make_unresolved_row(
    item: dict[str, Any],
    ref: dict[str, Any] | None,
    reason: str,
    ref_index: int | None = None,
) -> dict[str, Any]:
    ref = ref or {}
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "category": item.get("category"),
        "ref_index": ref_index,
        "reason": reason,
        "expected_act_title": ref.get("act_title") or item.get("expected_act_title"),
        "expected_canonical_key": ref.get("canonical_key") or item.get("expected_canonical_key"),
        "expected_article_no": ref.get("article_no") or item.get("expected_article_no"),
        "gold_notes": item.get("gold_notes"),
    }


def write_unresolved_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "category",
        "ref_index",
        "reason",
        "expected_act_title",
        "expected_canonical_key",
        "expected_article_no",
        "gold_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    if not args.db_url:
        raise SystemExit("[ERROR] Нужен URL БД. Передайте --db-url или заполните .env.")
    if args.max_matches_per_ref <= 0:
        raise SystemExit("[ERROR] --max-matches-per-ref должен быть больше 0")

    items = read_jsonl(args.input)
    resolved, unresolved = resolve_items(
        db_url=args.db_url,
        items=items,
        allow_title_fallback=not args.no_title_fallback,
        max_matches_per_ref=args.max_matches_per_ref,
    )
    write_jsonl(args.output, resolved)
    write_unresolved_csv(args.unresolved, unresolved)

    status_counts: dict[str, int] = {}
    for item in resolved:
        status = str(item.get("resolution_status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"[OK] questions: {len(resolved)}")
    print(f"[OK] status_counts: {status_counts}")
    print(f"[OK] wrote: {args.output}")
    print(f"[OK] unresolved rows: {len(unresolved)} -> {args.unresolved}")


if __name__ == "__main__":
    main()
