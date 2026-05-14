from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, AppConfig, RetrievalConfig, load_config
from law_qa_rag.env import get_database_url
from law_qa_rag.retrieval import RetrievedChunk, retrieve_chunks


DEFAULT_INPUT = Path("eval/gold_resolved.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценить качество retrieval на размеченных вопросах")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--db-url", type=str, default=get_database_url(required=False))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default="sparse,dense,weighted_hybrid",
        help="Список методов через запятую: sparse,dense,weighted_hybrid",
    )
    parser.add_argument(
        "--hybrid-weights",
        type=str,
        default="0.4:0.6,0.5:0.5,0.3:0.7",
        help="Веса hybrid в формате sparse:dense,sparse:dense",
    )
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число вопросов для пробного запуска")
    parser.add_argument("--no-warmup", action="store_true", help="Не выполнять прогрев dense-модели")
    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Включить вопросы без gold_chunk_ids в подробный JSONL как skipped; в метрики они не входят.",
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


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"sparse", "dense", "weighted_hybrid"}
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"Неизвестные методы retrieval: {unknown}")
    return methods


def parse_hybrid_weights(value: str) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            sparse_raw, dense_raw = item.split(":", 1)
            sparse_weight = float(sparse_raw)
            dense_weight = float(dense_raw)
        except ValueError as exc:
            raise ValueError(f"Некорректный формат веса hybrid: {item!r}") from exc
        if sparse_weight < 0 or dense_weight < 0:
            raise ValueError("Веса hybrid должны быть неотрицательными")
        result.append((sparse_weight, dense_weight))
    if not result:
        result.append((0.4, 0.6))
    return result


def build_eval_configs(args: argparse.Namespace, config: AppConfig) -> list[tuple[str, RetrievalConfig]]:
    methods = parse_methods(args.methods)
    hybrid_weights = parse_hybrid_weights(args.hybrid_weights)
    base = config.retrieval
    common = {
        "top_k": args.top_k,
        "candidate_limit": args.candidate_limit,
        "rrf_k": args.rrf_k or base.rrf_k,
    }

    configs: list[tuple[str, RetrievalConfig]] = []
    if "sparse" in methods:
        configs.append(("sparse", replace(base, method="sparse", **common)))
    if "dense" in methods:
        configs.append(("dense", replace(base, method="dense", **common)))
    if "weighted_hybrid" in methods:
        for sparse_weight, dense_weight in hybrid_weights:
            name = f"weighted_hybrid_s{sparse_weight:g}_d{dense_weight:g}"
            configs.append(
                (
                    name,
                    replace(
                        base,
                        method="weighted_hybrid",
                        sparse_weight=sparse_weight,
                        dense_weight=dense_weight,
                        **common,
                    ),
                )
            )
    return configs


def chunk_to_result(chunk: RetrievedChunk, rank: int) -> dict[str, Any]:
    data = asdict(chunk)
    data["rank"] = rank
    # Полный текст сильно раздувает файл; для анализа достаточно метаданных и короткого фрагмента.
    full_text = data.pop("full_text", "") or ""
    data["text_preview"] = str(full_text)[:500].replace("\n", " ")
    return data


def gold_ids(item: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for value in item.get("gold_chunk_ids") or []:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(ids))


def compute_metrics(retrieved_ids: list[int], gold: list[int]) -> dict[str, Any]:
    gold_set = set(gold)
    ranks = [idx for idx, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in gold_set]
    first_hit_rank = min(ranks) if ranks else None

    def hit_at(k: int) -> float:
        return 1.0 if gold_set.intersection(retrieved_ids[:k]) else 0.0

    def recall_at(k: int) -> float:
        if not gold_set:
            return 0.0
        return len(gold_set.intersection(retrieved_ids[:k])) / len(gold_set)

    return {
        "hit_at_1": hit_at(1),
        "hit_at_5": hit_at(5),
        "hit_at_10": hit_at(10),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "first_hit_rank": first_hit_rank,
        "found_gold_count_at_10": len(gold_set.intersection(retrieved_ids[:10])),
        "gold_count": len(gold_set),
    }


def warmup_if_needed(
    db_url: str,
    config: AppConfig,
    eval_configs: list[tuple[str, RetrievalConfig]],
    device: str,
    sample_question: str,
) -> None:
    if not any(rc.method in {"dense", "weighted_hybrid"} for _, rc in eval_configs):
        return
    dense_config = next(rc for _, rc in eval_configs if rc.method in {"dense", "weighted_hybrid"})
    try:
        retrieve_chunks(
            db_url=db_url,
            question=sample_question,
            retrieval_config=dense_config,
            embedding_config=config.embedding,
            device=device,
        )
        print("[OK] dense model/database warmup completed")
    except Exception as exc:
        print(f"[WARN] warmup failed and will be ignored: {exc}")


def classify_error(metrics: dict[str, Any] | None, exception: str | None) -> str:
    if exception:
        return "exception"
    if metrics is None:
        return "skipped_no_gold"
    if metrics["hit_at_10"] == 0:
        return "miss_at_10"
    if metrics["hit_at_1"] == 0:
        return "top1_miss"
    if metrics["recall_at_10"] < 1.0:
        return "partial_recall_at_10"
    return ""


def write_detailed(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, detailed_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in detailed_rows:
        if row.get("status") != "ok" or not row.get("metrics"):
            continue
        by_config.setdefault(str(row["config_name"]), []).append(row)

    fieldnames = [
        "config_name",
        "query_count",
        "hit_at_1",
        "hit_at_5",
        "hit_at_10",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "avg_latency_ms",
        "median_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_name, rows in by_config.items():
            metric_names = ["hit_at_1", "hit_at_5", "hit_at_10", "recall_at_5", "recall_at_10", "mrr"]
            out: dict[str, Any] = {"config_name": config_name, "query_count": len(rows)}
            for metric_name in metric_names:
                out[metric_name] = mean(row["metrics"][metric_name] for row in rows)
            latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
            out["avg_latency_ms"] = mean(latencies)
            out["median_latency_ms"] = statistics.median(latencies) if latencies else 0.0
            writer.writerow({key: round(value, 6) if isinstance(value, float) else value for key, value in out.items()})


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_error_analysis(path: Path, detailed_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "category",
        "config_name",
        "error_type",
        "first_hit_rank",
        "gold_chunk_ids",
        "top1_chunk_id",
        "top1_act_title",
        "top1_article_no",
        "exception",
        "gold_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_rows:
            metrics = row.get("metrics")
            error_type = classify_error(metrics, row.get("exception"))
            if not error_type:
                continue
            retrieved = row.get("retrieved") or []
            top1 = retrieved[0] if retrieved else {}
            writer.writerow(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "category": row.get("category"),
                    "config_name": row.get("config_name"),
                    "error_type": error_type,
                    "first_hit_rank": metrics.get("first_hit_rank") if metrics else "",
                    "gold_chunk_ids": json.dumps(row.get("gold_chunk_ids") or [], ensure_ascii=False),
                    "top1_chunk_id": top1.get("chunk_id"),
                    "top1_act_title": top1.get("act_title"),
                    "top1_article_no": top1.get("article_no"),
                    "exception": row.get("exception") or "",
                    "gold_notes": row.get("gold_notes") or "",
                }
            )


def main() -> None:
    args = parse_args()
    if not args.db_url:
        raise SystemExit("[ERROR] Нужен URL БД. Передайте --db-url или заполните .env.")
    if args.top_k <= 0 or args.candidate_limit <= 0:
        raise SystemExit("[ERROR] --top-k и --candidate-limit должны быть больше 0")

    config = load_config(args.settings)
    eval_configs = build_eval_configs(args, config)
    items = read_jsonl(args.input)
    if args.limit is not None:
        items = items[: args.limit]

    if items and not args.no_warmup:
        warmup_if_needed(args.db_url, config, eval_configs, args.device, str(items[0].get("question") or "тестовый вопрос"))

    detailed_rows: list[dict[str, Any]] = []
    total_runs = 0
    for item in items:
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        gold = gold_ids(item)
        if not gold and not args.include_unresolved:
            continue

        for config_name, retrieval_config in eval_configs:
            total_runs += 1
            started = time.perf_counter()
            exception = None
            retrieved_chunks: list[RetrievedChunk] = []
            try:
                if gold:
                    retrieved_chunks = retrieve_chunks(
                        db_url=args.db_url,
                        question=question,
                        retrieval_config=retrieval_config,
                        embedding_config=config.embedding,
                        device=args.device,
                    )
            except Exception as exc:  # noqa: BLE001 - для эксперимента важно сохранить все ошибки
                exception = str(exc)
            latency_ms = int((time.perf_counter() - started) * 1000)

            retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]
            metrics = compute_metrics(retrieved_ids, gold) if gold and exception is None else None
            detailed_rows.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "category": item.get("category"),
                    "requires_multiple_chunks": item.get("requires_multiple_chunks"),
                    "needs_clarification_expected": item.get("needs_clarification_expected"),
                    "config_name": config_name,
                    "retrieval_config": asdict(retrieval_config),
                    "gold_chunk_ids": gold,
                    "gold_notes": item.get("gold_notes"),
                    "latency_ms": latency_ms,
                    "status": "ok" if exception is None and gold else "error" if exception else "skipped_no_gold",
                    "exception": exception,
                    "metrics": metrics,
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved": [chunk_to_result(chunk, rank) for rank, chunk in enumerate(retrieved_chunks, start=1)],
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.out_dir / "retrieval_runs.jsonl"
    summary_path = args.out_dir / "summary_metrics.csv"
    errors_path = args.out_dir / "error_analysis.csv"
    write_detailed(detailed_path, detailed_rows)
    write_summary(summary_path, detailed_rows)
    write_error_analysis(errors_path, detailed_rows)

    print(f"[OK] configs: {[name for name, _ in eval_configs]}")
    print(f"[OK] runs: {total_runs}")
    print(f"[OK] wrote: {detailed_path}")
    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {errors_path}")


if __name__ == "__main__":
    main()
