from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, AppConfig, load_config
from law_qa_rag.env import get_database_url
from law_qa_rag.generation import GeneratedAnswer, generate_answer
from law_qa_rag.llm.gigachat_client import (
    GigaChatProvider,
    ensure_gigachat_credentials,
    validate_model_available,
)


DEFAULT_INPUT = Path("eval/gold_resolved.jsonl")
DEFAULT_OUT_DIR = Path("eval/results/generation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запустить ручную проверку генерации ответа на подмножестве вопросов"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--db-url", type=str, default=get_database_url(required=False))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Список id через запятую; если задан, --limit не применяется",
    )
    parser.add_argument(
        "--retrieval-method", choices=["sparse", "dense", "weighted_hybrid"], default=None
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=None)
    parser.add_argument("--sparse-weight", type=float, default=None)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument(
        "--include-unresolved", action="store_true", help="Включать вопросы без gold_chunk_ids"
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


def gold_ids(item: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for value in item.get("gold_chunk_ids") or []:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(ids))


def select_items(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_unresolved:
        items = [item for item in items if gold_ids(item)]

    if args.ids:
        needed = [item.strip() for item in args.ids.split(",") if item.strip()]
        needed_set = set(needed)
        return [item for item in items if str(item.get("id")) in needed_set]

    if args.limit and args.limit > 0:
        return items[: args.limit]
    return items


def override_config(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates: dict[str, Any] = {}
    if args.retrieval_method is not None:
        updates["method"] = args.retrieval_method
    if args.top_k is not None:
        updates["top_k"] = args.top_k
    if args.candidate_limit is not None:
        updates["candidate_limit"] = args.candidate_limit
    if args.rrf_k is not None:
        updates["rrf_k"] = args.rrf_k
    if args.sparse_weight is not None:
        updates["sparse_weight"] = args.sparse_weight
    if args.dense_weight is not None:
        updates["dense_weight"] = args.dense_weight

    if not updates:
        return config
    return replace(config, retrieval=replace(config.retrieval, **updates))


def result_to_row(
    item: dict[str, Any], result: GeneratedAnswer, total_latency_ms: int
) -> dict[str, Any]:
    payload = result.to_dict()
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "category": item.get("category"),
        "gold_chunk_ids": gold_ids(item),
        "gold_notes": item.get("gold_notes"),
        "needs_clarification_expected": item.get("needs_clarification_expected"),
        "status": "ok",
        "exception": None,
        "total_latency_ms": total_latency_ms,
        "llm_latency_ms": payload.get("latency_ms"),
        "answer": payload.get("answer"),
        "used_chunk_ids": payload.get("used_chunk_ids"),
        "needs_clarification": payload.get("needs_clarification"),
        "retrieval_method": payload.get("retrieval_method"),
        "retrieved_chunk_ids": payload.get("retrieved_chunk_ids"),
        "dropped_chunk_ids": payload.get("dropped_chunk_ids"),
        "llm_model": payload.get("llm_model"),
        "prompt_version": payload.get("prompt_version"),
        "answer_citations": payload.get("answer_citations"),
    }


def exception_to_row(item: dict[str, Any], exc: Exception, total_latency_ms: int) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "category": item.get("category"),
        "gold_chunk_ids": gold_ids(item),
        "gold_notes": item.get("gold_notes"),
        "needs_clarification_expected": item.get("needs_clarification_expected"),
        "status": "error",
        "exception": str(exc),
        "total_latency_ms": total_latency_ms,
        "llm_latency_ms": None,
        "answer": None,
        "used_chunk_ids": [],
        "needs_clarification": None,
        "retrieval_method": None,
        "retrieved_chunk_ids": [],
        "dropped_chunk_ids": [],
        "llm_model": None,
        "prompt_version": None,
        "answer_citations": [],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manual_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "question",
        "gold_chunk_ids",
        "retrieved_chunk_ids",
        "used_chunk_ids",
        "needs_clarification_expected",
        "needs_clarification",
        "answer",
        "context_sufficient_manual",
        "answer_supported_by_citations_manual",
        "no_extra_claims_manual",
        "citation_relevance_manual",
        "legal_meaning_preserved_manual",
        "comment",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "question": row.get("question"),
                    "gold_chunk_ids": json.dumps(
                        row.get("gold_chunk_ids") or [], ensure_ascii=False
                    ),
                    "retrieved_chunk_ids": json.dumps(
                        row.get("retrieved_chunk_ids") or [], ensure_ascii=False
                    ),
                    "used_chunk_ids": json.dumps(
                        row.get("used_chunk_ids") or [], ensure_ascii=False
                    ),
                    "needs_clarification_expected": row.get("needs_clarification_expected"),
                    "needs_clarification": row.get("needs_clarification"),
                    "answer": row.get("answer") or "",
                    "context_sufficient_manual": "",
                    "answer_supported_by_citations_manual": "",
                    "no_extra_claims_manual": "",
                    "citation_relevance_manual": "",
                    "legal_meaning_preserved_manual": "",
                    "comment": row.get("exception") or "",
                }
            )


def write_config_snapshot(
    path: Path, args: argparse.Namespace, config: AppConfig, question_count: int
) -> None:
    """Сохраняет YAML snapshot параметров generation-эксперимента."""
    payload = {
        "experiment": {
            "task": "answer_generation",
            "question_count": question_count,
            "input_file": str(args.input),
            "settings_file": str(args.settings),
        },
        "embedding": asdict(config.embedding),
        "retrieval": asdict(config.retrieval),
        "llm": asdict(config.llm),
        "outputs": {
            "detailed_runs": str(args.out_dir / "generation_all_v1.jsonl"),
            "manual_review": str(args.out_dir / "generation_all_v1.csv"),
        },
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.db_url:
        raise SystemExit("[ERROR] Нужен URL БД. Передайте --db-url или заполните .env.")

    config = override_config(load_config(args.settings), args)
    items = select_items(read_jsonl(args.input), args)
    if not items:
        raise SystemExit(
            "[ERROR] Нет вопросов для проверки генерации. Сначала выполните resolve_gold_chunks.py."
        )

    ensure_gigachat_credentials()
    provider = GigaChatProvider(model=config.llm.model)
    validate_model_available(provider, config.llm.model)

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        print(f"[INFO] {idx}/{len(items)} {item.get('id')}: {question}")
        started = time.perf_counter()
        try:
            result = generate_answer(
                question=question,
                db_url=args.db_url,
                config=config,
                provider=provider,
                device=args.device,
            )
            total_latency_ms = int((time.perf_counter() - started) * 1000)
            rows.append(result_to_row(item, result, total_latency_ms))
        except Exception as exc:  # noqa: BLE001 - эксперимент должен сохранять ошибку и идти дальше
            total_latency_ms = int((time.perf_counter() - started) * 1000)
            rows.append(exception_to_row(item, exc, total_latency_ms))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.out_dir / "generation_all_v1.jsonl"
    review_path = args.out_dir / "generation_all_v1.csv"
    snapshot_path = args.out_dir / "config_snapshot.yaml"
    write_jsonl(runs_path, rows)
    write_manual_review_csv(review_path, rows)
    write_config_snapshot(snapshot_path, args, config, len(items))

    print(f"[OK] wrote: {runs_path}")
    print(f"[OK] wrote: {review_path}")
    print(f"[OK] wrote: {snapshot_path}")


if __name__ == "__main__":
    main()
