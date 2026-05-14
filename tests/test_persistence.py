from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from law_qa_rag.generation import AnswerCitation, GeneratedAnswer
from law_qa_rag.persistence import (
    build_fragment_title,
    build_source_citation_label,
    clean_display_quote,
    ensure_technical_user,
    format_ru_date,
    format_status_label,
    get_feedback_for_answer_and_user,
    get_user_question_history,
    get_user_by_id,
    hash_password,
    normalize_question,
    save_answer_run_in_conn,
    save_feedback,
    update_last_login,
    verify_password,
)


class FakeCursor:
    def __init__(self, fetchone_values: list[Any], fetchall_values: list[Any] | None = None) -> None:
        self.fetchone_values = fetchone_values
        self.fetchall_values = fetchall_values or []
        self.executed: list[tuple[str, Any]] = []
        self.executemany_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: Any = None) -> None:
        self.executed.append((query, params))

    def executemany(self, query: str, params_seq: list[dict[str, Any]]) -> None:
        self.executemany_calls.append((query, params_seq))

    def fetchone(self) -> Any:
        return self.fetchone_values.pop(0)

    def fetchall(self) -> Any:
        return self.fetchall_values.pop(0)


class FakeConnection:
    def __init__(self, fetchone_values: list[Any], fetchall_values: list[Any] | None = None) -> None:
        self.cursor_obj = FakeCursor(fetchone_values, fetchall_values=fetchall_values)
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def make_generated_answer(needs_clarification: bool = False) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer="Ответ по контексту",
        used_chunk_ids=[101],
        needs_clarification=needs_clarification,
        answer_citations=[
            AnswerCitation(
                chunk_id=101,
                rank=1,
                relevance_score=0.42,
                quote="Полный текст chunk",
                act_title="Тестовый акт",
                doc_number="1-ФЗ",
                doc_date="2024-01-01",
                structure_ref="Статья 1",
                article_no="1",
                clause_range=None,
            )
        ],
        retrieval_method="weighted_hybrid",
        retrieved_chunk_ids=[101, 102],
        dropped_chunk_ids=[103],
        llm_model=None,
        prompt_version="answer_v1",
        latency_ms=1234,
    )


class PersistenceTests(unittest.TestCase):
    def test_password_hash_verification(self) -> None:
        password_hash = hash_password("secret")

        self.assertTrue(verify_password("secret", password_hash))
        self.assertFalse(verify_password("wrong", password_hash))
        self.assertNotIn("secret", password_hash)

    def test_normalize_question_collapses_spaces_and_lowercases(self) -> None:
        self.assertEqual(
            normalize_question("  Что   Такое ВОДНЫЙ объект? "),
            "что такое водный объект?",
        )

    def test_clean_display_quote_removes_repeated_act_and_structure(self) -> None:
        text = (
            "Лесной кодекс Российской Федерации "
            "Глава 3. Охрана лесов от пожаров Статья 53.5. "
            "Органы государственной власти, органы местного самоуправления "
            "обеспечивают меры пожарной безопасности."
        )

        self.assertEqual(
            clean_display_quote(
                text,
                act_title="Лесной кодекс Российской Федерации",
                structure_ref="Глава 3. Охрана лесов от пожаров Статья 53.5.",
            ),
            (
                "Органы государственной власти, органы местного самоуправления "
                "обеспечивают меры пожарной безопасности."
            ),
        )

    def test_clean_display_quote_keeps_text_when_prefix_boundary_differs(self) -> None:
        self.assertEqual(
            clean_display_quote("Статья 10 регулирует порядок.", structure_ref="Статья 1"),
            "Статья 10 регулирует порядок.",
        )

    def test_status_and_date_display_helpers(self) -> None:
        self.assertEqual(format_status_label("actual"), "действует")
        self.assertEqual(
            format_status_label("actual_with_future_editions"),
            "действует, есть будущие редакции",
        )
        self.assertEqual(format_status_label("legacy"), "не определен (legacy)")
        self.assertEqual(format_ru_date("2026-03-01"), "01.03.2026")

    def test_fragment_and_citation_labels_are_human_readable(self) -> None:
        chunk = {
            "chunk_index": 90,
            "structure_ref": "Глава 3. Статья 53.5",
            "article_no": "53.5",
        }

        self.assertEqual(build_fragment_title(chunk), "Глава 3. Статья 53.5")
        self.assertEqual(
            build_source_citation_label(1, chunk),
            "Цитата 1 — статья 53.5",
        )

    def test_ensure_technical_user_is_idempotent(self) -> None:
        conn = FakeConnection([(7,)])

        user_id = ensure_technical_user(conn, "local-web")

        self.assertEqual(user_id, 7)
        query = conn.cursor_obj.executed[0][0]
        self.assertIn("ON CONFLICT (external_uid)", query)

    def test_get_user_by_id_returns_user_row(self) -> None:
        row = {
            "id": 5,
            "external_uid": "user",
            "password_hash": "hash",
            "display_name": "User",
            "last_login_at": None,
            "created_at": "now",
        }
        conn = FakeConnection([row])

        with patch("law_qa_rag.persistence.psycopg.connect", return_value=conn):
            user = get_user_by_id("postgresql://test", 5)

        self.assertEqual(user["external_uid"], "user")
        query, params = conn.cursor_obj.executed[0]
        self.assertIn("WHERE id = %(user_id)s", query)
        self.assertEqual(params["user_id"], 5)

    def test_update_last_login_returns_user_and_commits(self) -> None:
        row = {
            "id": 5,
            "external_uid": "user",
            "password_hash": "hash",
            "display_name": None,
            "last_login_at": "now",
            "created_at": "before",
        }
        conn = FakeConnection([row])

        with patch("law_qa_rag.persistence.psycopg.connect", return_value=conn):
            user = update_last_login("postgresql://test", 5)

        self.assertEqual(user["id"], 5)
        self.assertTrue(conn.committed)
        self.assertIn("SET last_login_at = now()", conn.cursor_obj.executed[0][0])

    def test_get_user_question_history_filters_by_user_and_limit(self) -> None:
        rows = [
            {
                "query_id": 10,
                "question": "Что проверить?",
                "question_created_at": "2026-05-14 10:00",
                "answer_id": 20,
                "answer_created_at": "2026-05-14 10:01",
                "needs_clarification": False,
                "citation_count": 2,
            }
        ]
        conn = FakeConnection([], fetchall_values=[rows])

        with patch("law_qa_rag.persistence.psycopg.connect", return_value=conn):
            history = get_user_question_history("postgresql://test", user_id=5, limit=25)

        self.assertEqual(history, rows)
        query, params = conn.cursor_obj.executed[0]
        self.assertIn("WHERE q.user_id = %(user_id)s", query)
        self.assertIn("LEFT JOIN answers", query)
        self.assertEqual(params["user_id"], 5)
        self.assertEqual(params["limit"], 25)

    def test_get_user_question_history_rejects_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            get_user_question_history("postgresql://test", user_id=5, limit=0)

    def test_save_answer_run_writes_query_answer_and_citations(self) -> None:
        conn = FakeConnection([(7,), (11,), (22,)])

        answer_id = save_answer_run_in_conn(
            conn,
            "  Что проверить?  ",
            make_generated_answer(needs_clarification=True),
        )

        self.assertEqual(answer_id, 22)
        self.assertFalse(
            any("ALTER TABLE answers" in query for query, _ in conn.cursor_obj.executed)
        )

        query_params = next(
            params
            for _query, params in conn.cursor_obj.executed
            if isinstance(params, dict) and "normalized_question" in params
        )
        self.assertEqual(query_params["normalized_question"], "что проверить?")

        answer_params = next(
            params
            for _query, params in conn.cursor_obj.executed
            if isinstance(params, dict) and "latency_ms" in params
        )
        self.assertTrue(answer_params["needs_clarification"])
        self.assertEqual(answer_params["llm_model"], "sdk_default")
        self.assertEqual(answer_params["retrieval_method"], "weighted_hybrid")
        self.assertEqual(answer_params["latency_ms"], 1234)

        self.assertEqual(len(conn.cursor_obj.executemany_calls), 1)
        citation_rows = conn.cursor_obj.executemany_calls[0][1]
        self.assertEqual(citation_rows[0]["answer_id"], 22)
        self.assertEqual(citation_rows[0]["chunk_id"], 101)
        self.assertEqual(citation_rows[0]["quote"], "Полный текст chunk")

    def test_save_answer_run_uses_explicit_user_id(self) -> None:
        conn = FakeConnection([(11,), (22,)])

        answer_id = save_answer_run_in_conn(
            conn,
            "Что проверить?",
            make_generated_answer(),
            user_id=99,
        )

        self.assertEqual(answer_id, 22)
        self.assertFalse(
            any("INSERT INTO users" in query for query, _ in conn.cursor_obj.executed)
        )
        query_params = next(
            params
            for _query, params in conn.cursor_obj.executed
            if isinstance(params, dict) and "normalized_question" in params
        )
        self.assertEqual(query_params["user_id"], 99)

    def test_save_feedback_uses_upsert(self) -> None:
        conn = FakeConnection([(55,)])

        with patch("law_qa_rag.persistence.psycopg.connect", return_value=conn):
            feedback_id = save_feedback(
                "postgresql://test",
                answer_id=10,
                user_id=20,
                rating=5,
                comment=" Полезно ",
            )

        self.assertEqual(feedback_id, 55)
        self.assertTrue(conn.committed)
        query, params = conn.cursor_obj.executed[0]
        self.assertIn("ON CONFLICT (answer_id, user_id)", query)
        self.assertEqual(params["rating"], 5)
        self.assertEqual(params["comment"], "Полезно")

    def test_save_feedback_rejects_invalid_rating(self) -> None:
        with self.assertRaises(ValueError):
            save_feedback("postgresql://test", answer_id=10, user_id=20, rating=0, comment=None)

    def test_get_feedback_for_answer_and_user_returns_row(self) -> None:
        row = {
            "id": 55,
            "answer_id": 10,
            "user_id": 20,
            "rating": 4,
            "comment": "ok",
            "created_at": "now",
        }
        conn = FakeConnection([row])

        with patch("law_qa_rag.persistence.psycopg.connect", return_value=conn):
            feedback = get_feedback_for_answer_and_user(
                "postgresql://test",
                answer_id=10,
                user_id=20,
            )

        self.assertEqual(feedback["rating"], 4)
        query, params = conn.cursor_obj.executed[0]
        self.assertIn("FROM feedback", query)
        self.assertEqual(params["answer_id"], 10)
        self.assertEqual(params["user_id"], 20)


if __name__ == "__main__":
    unittest.main()
