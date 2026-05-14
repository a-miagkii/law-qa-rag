from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import date, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from law_qa_rag.generation import AnswerCitation, GeneratedAnswer


TECHNICAL_USER_UID = "local-web"
DEFAULT_LLM_MODEL_NAME = "sdk_default"
PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 210_000
PASSWORD_SALT_BYTES = 16
STATUS_LABELS = {
    "actual": "действует",
    "actual_with_future_editions": "действует, есть будущие редакции",
    "inactive": "утратил силу",
    "unknown": "не определен",
}


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


def hash_password(password: str) -> str:
    """Хеширует пароль через PBKDF2-SHA256."""
    if not password:
        raise ValueError("Пароль не должен быть пустым")
    salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return (
        f"{PASSWORD_HASH_ALGORITHM}${PASSWORD_HASH_ITERATIONS}"
        f"${salt.hex()}${digest.hex()}"
    )


def verify_password(password: str, password_hash: str | None) -> bool:
    """Проверяет пароль против PBKDF2-хеша."""
    if not password or not password_hash:
        return False
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = password_hash.split("$", 3)
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False
    if algorithm != PASSWORD_HASH_ALGORITHM:
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def get_user_by_external_uid(db_url: str, external_uid: str) -> dict[str, Any] | None:
    """Ищет пользователя по external_uid."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, external_uid, password_hash, display_name, last_login_at, created_at
                FROM users
                WHERE external_uid = %(external_uid)s;
                """,
                {"external_uid": external_uid},
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def get_user_by_id(db_url: str, user_id: int) -> dict[str, Any] | None:
    """Ищет пользователя по id."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, external_uid, password_hash, display_name, last_login_at, created_at
                FROM users
                WHERE id = %(user_id)s;
                """,
                {"user_id": user_id},
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def create_user(
    db_url: str,
    external_uid: str,
    password: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Создает локального пользователя с password_hash."""
    external_uid = normalize_login(external_uid)
    display_name = normalize_display_name(display_name)
    if not external_uid:
        raise ValueError("Введите логин")
    password_hash = hash_password(password)

    try:
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (external_uid, password_hash, display_name)
                    VALUES (%(external_uid)s, %(password_hash)s, %(display_name)s)
                    RETURNING id, external_uid, password_hash, display_name, last_login_at, created_at;
                    """,
                    {
                        "external_uid": external_uid,
                        "password_hash": password_hash,
                        "display_name": display_name,
                    },
                )
                row = cur.fetchone()
            conn.commit()
    except psycopg.errors.UniqueViolation as exc:
        raise ValueError("Пользователь с таким логином уже существует") from exc

    if row is None:
        raise RuntimeError("Не удалось создать пользователя")
    return dict(row)


def authenticate_user(
    db_url: str,
    external_uid: str,
    password: str,
) -> dict[str, Any] | None:
    """Проверяет логин/пароль и обновляет last_login_at."""
    external_uid = normalize_login(external_uid)
    user = get_user_by_external_uid(db_url, external_uid)
    if user is None or not verify_password(password, user.get("password_hash")):
        return None

    return update_last_login(db_url, int(user["id"]))


def update_last_login(db_url: str, user_id: int) -> dict[str, Any]:
    """Обновляет last_login_at и возвращает пользователя."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET last_login_at = now()
                WHERE id = %(user_id)s
                RETURNING id, external_uid, password_hash, display_name, last_login_at, created_at;
                """,
                {"user_id": user_id},
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        raise LookupError(f"Пользователь не найден: {user_id}")
    return dict(row)


def normalize_login(value: str | None) -> str:
    """Нормализует логин пользователя."""
    return str(value or "").strip().lower()


def normalize_display_name(value: str | None) -> str | None:
    """Нормализует отображаемое имя."""
    normalized = str(value or "").strip()
    return normalized or None


def clean_display_quote(
    text: str | None,
    act_title: str | None = None,
    structure_ref: str | None = None,
) -> str:
    """Убирает из display-текста уже показанные реквизиты акта и структуры."""
    original = str(text or "").strip()
    if not original:
        return original

    candidates = _display_prefix_candidates(act_title, structure_ref)
    cleaned = original
    for _ in range(3):
        before = cleaned
        for candidate in candidates:
            cleaned = _strip_display_prefix(cleaned, candidate)
        cleaned = _strip_redundant_display_lines(cleaned, candidates)
        if cleaned == before:
            break

    cleaned = _normalize_display_newlines(cleaned)
    return cleaned or original


def format_status_label(raw_status: Any) -> str:
    """Возвращает пользовательскую подпись статуса акта."""
    if raw_status is None:
        return STATUS_LABELS["unknown"]
    normalized = str(raw_status).strip()
    if not normalized:
        return STATUS_LABELS["unknown"]
    return STATUS_LABELS.get(normalized, f"не определен ({normalized})")


def format_ru_date(value: Any) -> str:
    """Форматирует дату как DD.MM.YYYY, если значение похоже на дату."""
    if value is None:
        return "не указана"
    if isinstance(value, datetime):
        return value.date().strftime("%d.%m.%Y")
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")

    raw = str(value).strip()
    if not raw:
        return "не указана"
    try:
        return date.fromisoformat(raw[:10]).strftime("%d.%m.%Y")
    except ValueError:
        return raw


def build_fragment_title(chunk: dict[str, Any]) -> str:
    """Строит юридический заголовок фрагмента без технического слова chunk."""
    structure_ref = str(chunk.get("structure_ref") or "").strip()
    if structure_ref:
        return structure_ref
    article_no = str(chunk.get("article_no") or "").strip()
    if article_no:
        return f"Статья {article_no}"
    return f"Фрагмент {chunk.get('chunk_index')}"


def build_source_citation_label(display_index: int, chunk: dict[str, Any]) -> str:
    """Строит подпись ссылки на цитату в источнике."""
    label = f"Цитата {display_index}"
    article_no = str(chunk.get("article_no") or "").strip()
    if article_no:
        return f"{label} — статья {article_no}"
    structure_ref = str(chunk.get("structure_ref") or "").strip()
    if structure_ref:
        return f"{label} — {_shorten_label(structure_ref)}"
    return label


def save_answer_run(
    db_url: str,
    question: str,
    result: GeneratedAnswer,
    external_uid: str = TECHNICAL_USER_UID,
    user_id: int | None = None,
) -> int:
    """Сохраняет query, answer и citations в PostgreSQL."""
    with psycopg.connect(db_url) as conn:
        answer_id = save_answer_run_in_conn(
            conn,
            question,
            result,
            external_uid=external_uid,
            user_id=user_id,
        )
        conn.commit()
        return answer_id


def save_answer_run_in_conn(
    conn: psycopg.Connection,
    question: str,
    result: GeneratedAnswer,
    external_uid: str = TECHNICAL_USER_UID,
    user_id: int | None = None,
) -> int:
    """Сохраняет результат генерации в уже открытом соединении."""
    query_user_id = user_id if user_id is not None else ensure_technical_user(conn, external_uid)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO queries (user_id, question, normalized_question)
            VALUES (%(user_id)s, %(question)s, %(normalized_question)s)
            RETURNING id;
            """,
            {
                "user_id": query_user_id,
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


def save_feedback(
    db_url: str,
    answer_id: int,
    user_id: int,
    rating: int,
    comment: str | None,
) -> int:
    """Сохраняет или обновляет пользовательскую оценку ответа."""
    if rating < 1 or rating > 5:
        raise ValueError("Оценка должна быть от 1 до 5")
    normalized_comment = str(comment or "").strip() or None

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (answer_id, user_id, rating, comment)
                VALUES (%(answer_id)s, %(user_id)s, %(rating)s, %(comment)s)
                ON CONFLICT (answer_id, user_id)
                DO UPDATE SET
                    rating = EXCLUDED.rating,
                    comment = EXCLUDED.comment,
                    created_at = now()
                RETURNING id;
                """,
                {
                    "answer_id": answer_id,
                    "user_id": user_id,
                    "rating": rating,
                    "comment": normalized_comment,
                },
            )
            feedback_id = int(_row_value(cur.fetchone(), "id"))
        conn.commit()
    return feedback_id


def get_feedback_for_answer_and_user(
    db_url: str,
    answer_id: int,
    user_id: int,
) -> dict[str, Any] | None:
    """Загружает feedback пользователя по конкретному ответу."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, answer_id, user_id, rating, comment, created_at
                FROM feedback
                WHERE answer_id = %(answer_id)s
                  AND user_id = %(user_id)s;
                """,
                {"answer_id": answer_id, "user_id": user_id},
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def get_user_question_history(
    db_url: str,
    user_id: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Загружает историю вопросов и ответов пользователя."""
    if limit < 1:
        raise ValueError("limit должен быть положительным")

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    q.id AS query_id,
                    q.question,
                    q.created_at AS question_created_at,
                    a.id AS answer_id,
                    a.created_at AS answer_created_at,
                    a.needs_clarification,
                    COALESCE(c.citation_count, 0) AS citation_count
                FROM queries q
                LEFT JOIN answers a ON a.query_id = q.id
                LEFT JOIN (
                    SELECT answer_id, count(*) AS citation_count
                    FROM answer_citations
                    GROUP BY answer_id
                ) c ON c.answer_id = a.id
                WHERE q.user_id = %(user_id)s
                ORDER BY q.created_at DESC, a.created_at DESC
                LIMIT %(limit)s;
                """,
                {"user_id": user_id, "limit": limit},
            )
            rows = cur.fetchall()
    return [dict(row) for row in rows]


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
                    q.user_id,
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

    for citation in citations:
        citation["display_quote"] = clean_display_quote(
            citation.get("quote"),
            act_title=citation.get("act_title"),
            structure_ref=citation.get("structure_ref"),
        )
        citation["doc_date_label"] = format_ru_date(citation.get("doc_date"))

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
                chunk["display_title"] = build_fragment_title(chunk)
                chunk["display_text"] = clean_display_quote(
                    chunk.get("text"),
                    act_title=act.get("title"),
                    structure_ref=chunk.get("structure_ref"),
                )
                chunk["display_citation_quote"] = (
                    clean_display_quote(
                        citation["quote"],
                        act_title=act.get("title"),
                        structure_ref=chunk.get("structure_ref"),
                    )
                    if citation
                    else None
                )
                chunks.append(chunk)

    act_dict = dict(act)
    act_dict["doc_date_label"] = format_ru_date(act_dict.get("doc_date"))
    act_dict["edition_as_of_label"] = format_ru_date(act_dict.get("edition_as_of"))
    act_dict["status_label"] = format_status_label(act_dict.get("status"))
    act_dict["show_edition_note"] = should_show_edition_note(act_dict)

    cited_chunks = sorted(
        (chunk for chunk in chunks if chunk["is_cited"]),
        key=lambda chunk: (chunk["citation_rank"] or 0, chunk["chunk_id"]),
    )
    source_citations: list[dict[str, Any]] = []
    for display_index, chunk in enumerate(cited_chunks, start=1):
        chunk["citation_display_index"] = display_index
        source_citations.append(
            {
                "chunk_id": chunk["chunk_id"],
                "display_index": display_index,
                "label": build_source_citation_label(display_index, chunk),
            }
        )

    return {
        "act": act_dict,
        "chunks": chunks,
        "source_citations": source_citations,
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


def should_show_edition_note(act: dict[str, Any]) -> bool:
    """Определяет, нужно ли показывать примечание о редакции в UI."""
    note = str(act.get("edition_note") or "").strip()
    if not note:
        return False

    normalized_note = _normalize_display_text(note)
    duplicates = [
        act.get("status"),
        act.get("status_label"),
        act.get("edition_as_of"),
        act.get("edition_as_of_label"),
    ]
    normalized_duplicates = {
        _normalize_display_text(value)
        for value in duplicates
        if value is not None and str(value).strip()
    }
    normalized_duplicates |= {
        f"редакция {value}"
        for value in normalized_duplicates
    }
    if normalized_note in normalized_duplicates:
        return False

    edition_label = _normalize_display_text(act.get("edition_as_of_label"))
    status_label = _normalize_display_text(act.get("status_label"))
    if edition_label and edition_label in normalized_note and len(normalized_note) <= len(edition_label) + 24:
        return False
    if status_label and status_label in normalized_note and len(normalized_note) <= len(status_label) + 24:
        return False
    return True


def _display_prefix_candidates(
    act_title: str | None,
    structure_ref: str | None,
) -> list[str]:
    """Собирает возможные повторяющиеся префиксы для display-очистки."""
    candidates: list[str] = []
    for value in (act_title, structure_ref):
        normalized = str(value or "").strip()
        if normalized:
            candidates.append(normalized)

    if structure_ref:
        for part in re.split(r"\s*(?:[>/|]+|\n+)\s*", str(structure_ref)):
            part = part.strip()
            if part and part not in candidates:
                candidates.append(part)
    return candidates


def _strip_display_prefix(text: str, prefix: str) -> str:
    """Снимает один известный префикс с начала текста."""
    text = text.strip()
    prefix = prefix.strip()
    if not text or not prefix:
        return text
    if _starts_with_display_prefix(text, prefix):
        return text[len(prefix):].lstrip(" \t\r\n:;.,-—–")

    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"(?=$|[\s:;.,\-—–])\s*[:;.,\-—–]*\s*"
    stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
    return stripped.strip() if stripped != text else text


def _strip_redundant_display_lines(text: str, candidates: list[str]) -> str:
    """Удаляет первые строки, если они дублируют реквизиты или структуру."""
    lines = [line.strip() for line in text.splitlines()]
    candidate_set = {
        _normalize_display_text(candidate)
        for candidate in candidates
        if candidate.strip()
    }
    while lines:
        line = lines[0]
        normalized_line = _normalize_display_text(line)
        if not normalized_line:
            lines.pop(0)
            continue
        if any(
            normalized_line == candidate
            or _starts_with_display_prefix(candidate, normalized_line)
            or _starts_with_display_prefix(normalized_line, candidate)
            for candidate in candidate_set
        ):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


def _normalize_display_newlines(text: str) -> str:
    """Убирает лишние пробелы вокруг переносов строк."""
    return "\n".join(line.strip() for line in text.strip().splitlines()).strip()


def _starts_with_display_prefix(text: str, prefix: str) -> bool:
    """Проверяет prefix с учетом границы слова/пунктуации."""
    text_folded = text.casefold()
    prefix_folded = prefix.casefold()
    if not text_folded.startswith(prefix_folded):
        return False
    if len(text_folded) == len(prefix_folded):
        return True
    return text_folded[len(prefix_folded)] in " \t\r\n:;.,-—–"


def _normalize_display_text(value: Any) -> str:
    """Нормализует display-текст для сравнения дублей."""
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _shorten_label(value: str, max_length: int = 72) -> str:
    """Сокращает длинную подпись для боковой панели."""
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _row_value(row: Any, key: str, index: int = 0) -> Any:
    """Достает значение из dict_row или tuple row."""
    if row is None:
        raise RuntimeError("DB query did not return a row")
    if isinstance(row, dict):
        return row[key]
    return row[index]
