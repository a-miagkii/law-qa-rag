from __future__ import annotations

import unittest
from typing import Any

from law_qa_rag.generation import AnswerCitation, GeneratedAnswer
from law_qa_rag.persistence import (
    ensure_technical_user,
    normalize_question,
    save_answer_run_in_conn,
)


class FakeCursor:
    def __init__(self, fetchone_values: list[Any]) -> None:
        self.fetchone_values = fetchone_values
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


class FakeConnection:
    def __init__(self, fetchone_values: list[Any]) -> None:
        self.cursor_obj = FakeCursor(fetchone_values)

    def cursor(self) -> FakeCursor:
        return self.cursor_obj


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
    def test_normalize_question_collapses_spaces_and_lowercases(self) -> None:
        self.assertEqual(
            normalize_question("  Что   Такое ВОДНЫЙ объект? "),
            "что такое водный объект?",
        )

    def test_ensure_technical_user_is_idempotent(self) -> None:
        conn = FakeConnection([(7,)])

        user_id = ensure_technical_user(conn, "local-web")

        self.assertEqual(user_id, 7)
        query = conn.cursor_obj.executed[0][0]
        self.assertIn("ON CONFLICT (external_uid)", query)

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


if __name__ == "__main__":
    unittest.main()
