from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import torch
import yaml
from sentence_transformers import SentenceTransformer


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "settings.yaml"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024


SELECT_CHUNKS_SQL = """
SELECT id, text
FROM chunks
WHERE embedding IS NULL
ORDER BY id
LIMIT %(limit)s;
"""


UPDATE_EMBEDDING_SQL = """
UPDATE chunks
SET
    embedding = %(embedding)s::vector,
    embedding_model = %(embedding_model)s
WHERE id = %(chunk_id)s;
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
        description="Посчитать embeddings для chunks и записать их в PostgreSQL",
        add_help=False,
    )
    parser._optionals.title = "параметры"

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Показать справку и выйти",
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
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL URL. Можно также передать через DATABASE_URL.",
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
        "--batch-size",
        type=positive_int,
        default=8,
        help="Сколько chunks кодировать за один batch",
    )

    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Ограничить общее число chunks для тестового запуска",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Устройство для модели embeddings",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Загрузить модель и прочитать chunks, но не писать embeddings в БД",
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

    model = SentenceTransformer(model_name, device=device)

    return model


def embedding_to_pgvector(value: np.ndarray) -> str:
    """Преобразует numpy-вектор в текстовый формат pgvector."""
    return "[" + ",".join(f"{float(x):.8f}" for x in value) + "]"


def fetch_chunks(cur: psycopg.Cursor, batch_size: int) -> list[tuple[int, str]]:
    """Берет следующий batch chunks без embeddings."""
    cur.execute(SELECT_CHUNKS_SQL, {"limit": batch_size})
    return list(cur.fetchall())


def count_chunks_without_embeddings(cur: psycopg.Cursor) -> int:
    """Считает chunks, для которых embeddings еще не записаны."""
    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL;")
    return int(cur.fetchone()[0])


def print_embedding_stats(cur: psycopg.Cursor) -> None:
    """Печатает текущую статистику embeddings в БД."""
    cur.execute("SELECT count(*) FROM chunks;")
    total = int(cur.fetchone()[0])

    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL;")
    without_embeddings = int(cur.fetchone()[0])

    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL;")
    with_embeddings = int(cur.fetchone()[0])

    print(f"[OK] chunks всего: {total}")
    print(f"[OK] chunks с embeddings: {with_embeddings}")
    print(f"[OK] chunks без embeddings: {without_embeddings}")

    cur.execute("""
        SELECT embedding_model, count(*)
        FROM chunks
        WHERE embedding IS NOT NULL
        GROUP BY embedding_model
        ORDER BY count(*) DESC;
    """)

    rows = cur.fetchall()
    if rows:
        print("[OK] embeddings по моделям:")
        for model_name, count in rows:
            print(f"  {count:>6}  {model_name}")


def format_duration(seconds: float) -> str:
    """Форматирует длительность в компактный человекочитаемый вид."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}с"

    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}м {sec:02d}с"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}ч {minutes:02d}м"


def estimate_eta_seconds(
    processed: int,
    target_total: int,
    elapsed_seconds: float,
) -> float | None:
    """Оценивает оставшееся время по средней скорости обработки."""
    if processed <= 0:
        return None

    remaining = max(0, target_total - processed)
    seconds_per_chunk = elapsed_seconds / processed
    return remaining * seconds_per_chunk


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    expected_dim: int,
) -> np.ndarray:
    """Кодирует тексты и проверяет размерность embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

    if embeddings.shape[1] != expected_dim:
        raise RuntimeError(
            f"Ожидалась размерность embeddings {expected_dim}, получено {embeddings.shape[1]}"
        )

    return embeddings


def update_embeddings(
    cur: psycopg.Cursor,
    chunk_ids: list[int],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    """Записывает embeddings для batch chunks."""
    rows: list[dict[str, Any]] = []

    for chunk_id, embedding in zip(chunk_ids, embeddings, strict=True):
        rows.append(
            {
                "chunk_id": chunk_id,
                "embedding": embedding_to_pgvector(embedding),
                "embedding_model": model_name,
            }
        )

    cur.executemany(UPDATE_EMBEDDING_SQL, rows)


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()

    if not args.db_url:
        raise ValueError(
            "Нужен URL БД. Передайте --db-url или задайте DATABASE_URL."
        )

    settings = read_settings(args.settings)
    config = build_embedding_config(settings, args.settings)
    if args.model is not None:
        config = replace(config, model_name=args.model)
    if args.embedding_dim is not None:
        config = replace(config, embedding_dim=args.embedding_dim)

    print(f"[OK] settings: {config.settings_path or 'не используются'}")
    print(f"[OK] embedding dim: {config.embedding_dim}")

    device = detect_device(args.device)
    model = load_model(config.model_name, device)

    total_processed = 0
    run_started_at = time.perf_counter()

    with psycopg.connect(args.db_url) as conn:
        with conn.cursor() as cur:
            print_embedding_stats(cur)

            initial_remaining = count_chunks_without_embeddings(cur)
            if initial_remaining == 0:
                print("[OK] нет chunks для расчета embeddings")
                return

            target_total = min(initial_remaining, args.limit) if args.limit else initial_remaining
            print(f"[OK] chunks к обработке: {target_total} из {initial_remaining}")

            while True:
                if args.limit is not None:
                    remaining_limit = args.limit - total_processed
                    if remaining_limit <= 0:
                        print(f"[OK] достигнут limit: {args.limit}")
                        break
                    current_batch_size = min(args.batch_size, remaining_limit)
                else:
                    current_batch_size = args.batch_size

                rows = fetch_chunks(cur, current_batch_size)

                if not rows:
                    print("[OK] больше нет chunks без embeddings")
                    break

                chunk_ids = [row[0] for row in rows]
                texts = [row[1] for row in rows]

                batch_started_at = time.perf_counter()
                embeddings = encode_texts(
                    model=model,
                    texts=texts,
                    batch_size=current_batch_size,
                    expected_dim=config.embedding_dim,
                )
                encode_elapsed = time.perf_counter() - batch_started_at

                if args.dry_run:
                    print(
                        f"[OK] dry-run batch: {len(chunk_ids)} chunks | "
                        f"shape={embeddings.shape} | "
                        f"расчет={format_duration(encode_elapsed)}"
                    )
                    break

                update_embeddings(
                    cur=cur,
                    chunk_ids=chunk_ids,
                    embeddings=embeddings,
                    model_name=config.model_name,
                )

                conn.commit()

                total_processed += len(chunk_ids)
                remaining = count_chunks_without_embeddings(cur)
                batch_elapsed = time.perf_counter() - batch_started_at
                total_elapsed = time.perf_counter() - run_started_at
                eta_seconds = estimate_eta_seconds(
                    processed=total_processed,
                    target_total=target_total,
                    elapsed_seconds=total_elapsed,
                )
                eta_text = format_duration(eta_seconds) if eta_seconds is not None else "неизвестно"

                print(
                    f"[OK] batch: {len(chunk_ids)} chunks | "
                    f"расчет={format_duration(encode_elapsed)} | "
                    f"batch_total={format_duration(batch_elapsed)} | "
                    f"обработано={total_processed}/{target_total} | "
                    f"осталось_в_БД={remaining} | "
                    f"ETA={eta_text}"
                )

            print_embedding_stats(cur)

    print("[OK] расчет embeddings завершен")


if __name__ == "__main__":
    main()
