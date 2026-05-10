from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from law_qa_rag.generation import AnswerCitation, GeneratedAnswer


TECHNICAL_USER_UID = "local-web"
DEFAULT_LLM_MODEL_NAME = "sdk_default"


def normalize_question(question: str) -> str:
    """Нормализует вопрос для сохранения в queries."""
    return " ".join(question.split()).lower()


def ensure_technical_user(
    conn: psycopg.Connection,
    external_uid: str = TECHNICAL_USER_UID,
) -> int:
    """Создает или возвращает технического пользователя web-прототипа."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (external_uid)
            VALUES (%(external_uid)s)
            ON CONFLICT (external_uid) DO UPDATE SET
                external_uid = EXCLUDED.external_uid
            RETURNING id;
            """,
            {"external_uid": external_uid},
        )
        return int(_row_value(cur.fetchone(), "id"))


def save_answer_run(
    db_url: str,
    question: str,
    result: GeneratedAnswer,
    external_uid: str = TECHNICAL_USER_UID,
) -> int:
    """Сохраняет query, answer и citations в PostgreSQL."""
    with psycopg.connect(db_url) as conn:
        answer_id = save_answer_run_in_conn(conn, question, result, external_uid)
        conn.commit()
        return answer_id


def save_answer_run_in_conn(
    conn: psycopg.Connection,
    question: str,
    result: GeneratedAnswer,
    external_uid: str = TECHNICAL_USER_UID,
) -> int:
    """Сохраняет результат генерации в уже открытом соединении."""
    user_id = ensure_technical_user(conn, external_uid)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO queries (user_id, question, normalized_question)
            VALUES (%(user_id)s, %(question)s, %(normalized_question)s)
            RETURNING id;
            """,
            {
                "user_id": user_id,
                "question": question,
                "normalized_question": normalize_question(question),
            },
        )
        query_id = int(_row_value(cur.fetchone(), "id"))

        cur.execute(
            """
            INSERT INTO answers (
                query_id,
                answer_text,
                llm_model,
                prompt_version,
                needs_clarification,
                retrieval_method,
                retrieved_chunk_ids,
                dropped_chunk_ids,
                latency_ms
            )
            VALUES (
                %(query_id)s,
                %(answer_text)s,
                %(llm_model)s,
                %(prompt_version)s,
                %(needs_clarification)s,
                %(retrieval_method)s,
                %(retrieved_chunk_ids)s,
                %(dropped_chunk_ids)s,
                %(latency_ms)s
            )
            RETURNING id;
            """,
            {
                "query_id": query_id,
                "answer_text": result.answer,
                "llm_model": result.llm_model or DEFAULT_LLM_MODEL_NAME,
                "prompt_version": result.prompt_version,
                "needs_clarification": result.needs_clarification,
                "retrieval_method": result.retrieval_method,
                "retrieved_chunk_ids": Jsonb(result.retrieved_chunk_ids),
                "dropped_chunk_ids": Jsonb(result.dropped_chunk_ids),
                "latency_ms": result.latency_ms,
            },
        )
        answer_id = int(_row_value(cur.fetchone(), "id"))

        citation_rows = [
            _citation_row(answer_id, citation)
            for citation in result.answer_citations
        ]
        if citation_rows:
            cur.executemany(
                """
                INSERT INTO answer_citations (
                    answer_id,
                    chunk_id,
                    rank,
                    relevance_score,
                    quote
                )
                VALUES (
                    %(answer_id)s,
                    %(chunk_id)s,
                    %(rank)s,
                    %(relevance_score)s,
                    %(quote)s
                );
                """,
                citation_rows,
            )

    return answer_id


def load_answer_page(db_url: str, answer_id: int) -> dict[str, Any]:
    """Загружает данные для страницы ответа."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ans.id AS answer_id,
                    ans.answer_text,
                    ans.llm_model,
                    ans.prompt_version,
                    ans.needs_clarification,
                    ans.retrieval_method,
                    ans.retrieved_chunk_ids,
                    ans.dropped_chunk_ids,
                    ans.latency_ms,
                    ans.created_at,
                    q.id AS query_id,
                    q.question,
                    q.normalized_question
                FROM answers ans
                JOIN queries q ON q.id = ans.query_id
                WHERE ans.id = %(answer_id)s;
                """,
                {"answer_id": answer_id},
            )
            answer = cur.fetchone()
            if answer is None:
                raise LookupError(f"Ответ не найден: {answer_id}")

            cur.execute(
                """
                SELECT
                    ac.id AS citation_id,
                    ac.chunk_id,
                    ac.rank,
                    ac.relevance_score,
                    ac.quote,
                    c.act_id,
                    c.chunk_index,
                    c.structure_ref,
                    c.article_no,
                    c.clause_range,
                    c.token_count,
                    a.title AS act_title,
                    a.doc_number,
                    a.doc_date,
                    a.edition_as_of,
                    a.edition_note,
                    a.status
                FROM answer_citations ac
                JOIN chunks c ON c.id = ac.chunk_id
                JOIN acts a ON a.id = c.act_id
                WHERE ac.answer_id = %(answer_id)s
                ORDER BY ac.rank, ac.id;
                """,
                {"answer_id": answer_id},
            )
            citations = [dict(row) for row in cur.fetchall()]

    return {"answer": dict(answer), "citations": citations}


def load_source_page(
    db_url: str,
    act_id: int,
    answer_id: int | None = None,
) -> dict[str, Any]:
    """Загружает акт и chunks для страницы источника."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
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
                    source_system,
                    imported_at
                FROM acts
                WHERE id = %(act_id)s;
                """,
                {"act_id": act_id},
            )
            act = cur.fetchone()
            if act is None:
                raise LookupError(f"Акт не найден: {act_id}")

            cited_by_chunk_id: dict[int, dict[str, Any]] = {}
            if answer_id is not None:
                cur.execute(
                    """
                    SELECT ac.chunk_id, ac.rank, ac.quote
                    FROM answer_citations ac
                    JOIN chunks c ON c.id = ac.chunk_id
                    WHERE ac.answer_id = %(answer_id)s
                      AND c.act_id = %(act_id)s
                    ORDER BY ac.rank, ac.id;
                    """,
                    {"answer_id": answer_id, "act_id": act_id},
                )
                cited_by_chunk_id = {
                    int(row["chunk_id"]): dict(row)
                    for row in cur.fetchall()
                }

            cur.execute(
                """
                SELECT
                    id AS chunk_id,
                    chunk_index,
                    text,
                    structure_ref,
                    article_no,
                    clause_range,
                    token_count
                FROM chunks
                WHERE act_id = %(act_id)s
                ORDER BY chunk_index;
                """,
                {"act_id": act_id},
            )
            chunks: list[dict[str, Any]] = []
            for row in cur.fetchall():
                chunk = dict(row)
                citation = cited_by_chunk_id.get(int(chunk["chunk_id"]))
                chunk["is_cited"] = citation is not None
                chunk["citation_rank"] = citation["rank"] if citation else None
                chunk["citation_quote"] = citation["quote"] if citation else None
                chunks.append(chunk)

    return {
        "act": dict(act),
        "chunks": chunks,
        "answer_id": answer_id,
        "highlighted_chunk_ids": sorted(cited_by_chunk_id),
    }


def _citation_row(answer_id: int, citation: AnswerCitation) -> dict[str, Any]:
    """Преобразует citation в параметры INSERT."""
    relevance_score = citation.relevance_score
    if relevance_score < 0:
        relevance_score = 0.0
    return {
        "answer_id": answer_id,
        "chunk_id": citation.chunk_id,
        "rank": citation.rank,
        "relevance_score": relevance_score,
        "quote": citation.quote,
    }


def _row_value(row: Any, key: str, index: int = 0) -> Any:
    """Достает значение из dict_row или tuple row."""
    if row is None:
        raise RuntimeError("DB query did not return a row")
    if isinstance(row, dict):
        return row[key]
    return row[index]
