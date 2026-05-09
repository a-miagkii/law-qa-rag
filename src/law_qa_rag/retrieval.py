from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row

from law_qa_rag.config import EmbeddingConfig, RetrievalConfig


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
    NULL::text AS embedding_model,
    ts_rank_cd(c.search_vector, q.query) AS sparse_score,
    NULL::double precision AS dense_score,
    NULL::double precision AS distance,
    c.text AS full_text
FROM chunks c
JOIN acts a ON a.id = c.act_id
CROSS JOIN q
WHERE c.search_vector @@ q.query
  AND (%(act_filter)s::text IS NULL OR a.title ILIKE '%%' || %(act_filter)s::text || '%%')
  AND (%(article_no)s::text IS NULL OR c.article_no = %(article_no)s::text)
ORDER BY sparse_score DESC, c.id
LIMIT %(candidate_limit)s;
"""


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
    NULL::double precision AS sparse_score,
    1 - (c.embedding <=> %(query_embedding)s::vector) AS dense_score,
    c.embedding <=> %(query_embedding)s::vector AS distance,
    c.text AS full_text
FROM chunks c
JOIN acts a ON a.id = c.act_id
WHERE c.embedding IS NOT NULL
  AND c.embedding_model = %(embedding_model)s::text
  AND (%(act_filter)s::text IS NULL OR a.title ILIKE '%%' || %(act_filter)s::text || '%%')
  AND (%(article_no)s::text IS NULL OR c.article_no = %(article_no)s::text)
ORDER BY c.embedding <=> %(query_embedding)s::vector ASC, c.id
LIMIT %(candidate_limit)s;
"""


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk, выбранный retrieval-этапом."""

    chunk_id: int
    act_id: int
    act_title: str
    doc_number: str | None
    doc_date: str | None
    chunk_index: int
    structure_ref: str | None
    article_no: str | None
    clause_range: str | None
    token_count: int
    full_text: str
    embedding_model: str | None = None
    sparse_rank: int | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    dense_score: float | None = None
    distance: float | None = None
    retrieval_score: float = 0.0


def detect_device(requested: str) -> str:
    """Выбирает устройство для embedding-модели."""
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedding_model(model_name: str, device: str) -> Any:
    """Загружает SentenceTransformer-модель."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embedding_to_pgvector(value: np.ndarray) -> str:
    """Преобразует numpy-вектор в текстовый формат pgvector."""
    return "[" + ",".join(f"{float(x):.8f}" for x in value) + "]"


def encode_query(model: Any, question: str, expected_dim: int) -> np.ndarray:
    """Кодирует запрос и проверяет размерность embedding."""
    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise RuntimeError(f"Некорректная shape query embedding: {embedding.shape}")
    if embedding.shape[1] != expected_dim:
        raise RuntimeError(
            f"Ожидалась размерность embeddings {expected_dim}, получено {embedding.shape[1]}"
        )
    return embedding[0]


def search_sparse(
    cur: psycopg.Cursor,
    question: str,
    candidate_limit: int,
    act_filter: str | None = None,
    article_no: str | None = None,
) -> list[dict[str, Any]]:
    """Выполняет PostgreSQL full-text search."""
    cur.execute(
        SPARSE_SEARCH_SQL,
        {
            "question": question,
            "candidate_limit": candidate_limit,
            "act_filter": act_filter,
            "article_no": article_no,
        },
    )
    return list(cur.fetchall())


def search_dense(
    cur: psycopg.Cursor,
    query_embedding: np.ndarray,
    embedding_model: str,
    candidate_limit: int,
    act_filter: str | None = None,
    article_no: str | None = None,
) -> list[dict[str, Any]]:
    """Выполняет pgvector dense search."""
    cur.execute(
        DENSE_SEARCH_SQL,
        {
            "query_embedding": embedding_to_pgvector(query_embedding),
            "embedding_model": embedding_model,
            "candidate_limit": candidate_limit,
            "act_filter": act_filter,
            "article_no": article_no,
        },
    )
    return list(cur.fetchall())


def row_to_chunk(
    row: dict[str, Any],
    sparse_rank: int | None = None,
    dense_rank: int | None = None,
    retrieval_score: float = 0.0,
) -> RetrievedChunk:
    """Преобразует DB row в RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=int(row["chunk_id"]),
        act_id=int(row["act_id"]),
        act_title=str(row["act_title"]),
        doc_number=str(row["doc_number"]) if row.get("doc_number") is not None else None,
        doc_date=str(row["doc_date"]) if row.get("doc_date") is not None else None,
        chunk_index=int(row["chunk_index"]),
        structure_ref=row.get("structure_ref"),
        article_no=row.get("article_no"),
        clause_range=row.get("clause_range"),
        token_count=int(row["token_count"]),
        full_text=str(row["full_text"]),
        embedding_model=row.get("embedding_model"),
        sparse_rank=sparse_rank,
        dense_rank=dense_rank,
        sparse_score=_optional_float(row.get("sparse_score")),
        dense_score=_optional_float(row.get("dense_score")),
        distance=_optional_float(row.get("distance")),
        retrieval_score=retrieval_score,
    )


def rank_sparse(rows: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """Проставляет sparse rank и score для sparse-only результатов."""
    chunks = []
    for rank, row in enumerate(rows, start=1):
        score = _optional_float(row.get("sparse_score")) or 0.0
        chunks.append(row_to_chunk(row, sparse_rank=rank, retrieval_score=score))
    return chunks


def rank_dense(rows: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """Проставляет dense rank и score для dense-only результатов."""
    chunks = []
    for rank, row in enumerate(rows, start=1):
        score = _optional_float(row.get("dense_score")) or 0.0
        chunks.append(row_to_chunk(row, dense_rank=rank, retrieval_score=score))
    return chunks


def weighted_rrf_fusion(
    sparse_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    """Сливает sparse и dense результаты через weighted RRF."""
    combined: dict[int, dict[str, Any]] = {}

    for rank, row in enumerate(sparse_rows, start=1):
        chunk_id = int(row["chunk_id"])
        if chunk_id not in combined:
            combined[chunk_id] = dict(row)
        combined[chunk_id]["sparse_rank"] = rank
        combined[chunk_id]["sparse_score"] = row.get("sparse_score")
        combined[chunk_id]["retrieval_score"] = combined[chunk_id].get("retrieval_score", 0.0)
        combined[chunk_id]["retrieval_score"] += config.sparse_weight / (config.rrf_k + rank)

    for rank, row in enumerate(dense_rows, start=1):
        chunk_id = int(row["chunk_id"])
        if chunk_id not in combined:
            combined[chunk_id] = dict(row)
        combined[chunk_id]["dense_rank"] = rank
        combined[chunk_id]["dense_score"] = row.get("dense_score")
        combined[chunk_id]["distance"] = row.get("distance")
        combined[chunk_id]["embedding_model"] = row.get("embedding_model")
        combined[chunk_id]["retrieval_score"] = combined[chunk_id].get("retrieval_score", 0.0)
        combined[chunk_id]["retrieval_score"] += config.dense_weight / (config.rrf_k + rank)

    chunks = [
        row_to_chunk(
            row,
            sparse_rank=row.get("sparse_rank"),
            dense_rank=row.get("dense_rank"),
            retrieval_score=float(row["retrieval_score"]),
        )
        for row in combined.values()
    ]
    return sorted(
        chunks,
        key=lambda item: (
            item.retrieval_score,
            -(item.sparse_rank or 10**9),
            -(item.dense_rank or 10**9),
        ),
        reverse=True,
    )


def retrieve_chunks(
    db_url: str,
    question: str,
    retrieval_config: RetrievalConfig,
    embedding_config: EmbeddingConfig,
    device: str = "auto",
    act_filter: str | None = None,
    article_no: str | None = None,
) -> list[RetrievedChunk]:
    """Выбирает chunks методом из RetrievalConfig."""
    method = retrieval_config.method
    query_embedding = None
    if method in {"dense", "weighted_hybrid"}:
        model = load_embedding_model(embedding_config.model_name, detect_device(device))
        query_embedding = encode_query(model, question, embedding_config.embedding_dim)

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            sparse_rows: list[dict[str, Any]] = []
            dense_rows: list[dict[str, Any]] = []
            if method in {"sparse", "weighted_hybrid"}:
                sparse_rows = search_sparse(
                    cur,
                    question,
                    retrieval_config.candidate_limit,
                    act_filter,
                    article_no,
                )
            if method in {"dense", "weighted_hybrid"}:
                if query_embedding is None:
                    raise RuntimeError("query_embedding не был рассчитан")
                dense_rows = search_dense(
                    cur,
                    query_embedding,
                    embedding_config.model_name,
                    retrieval_config.candidate_limit,
                    act_filter,
                    article_no,
                )

    if method == "sparse":
        return rank_sparse(sparse_rows)[: retrieval_config.top_k]
    if method == "dense":
        return rank_dense(dense_rows)[: retrieval_config.top_k]
    return weighted_rrf_fusion(sparse_rows, dense_rows, retrieval_config)[: retrieval_config.top_k]


def _optional_float(value: Any) -> float | None:
    """Преобразует nullable value в float."""
    if value is None:
        return None
    return float(value)
