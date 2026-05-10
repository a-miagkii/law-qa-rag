from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from law_qa_rag.generation import AnswerCitation, GeneratedAnswer
from law_qa_rag.web.app import app


def make_generated_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        answer="Водные объекты общего пользования доступны гражданам.",
        used_chunk_ids=[101],
        needs_clarification=False,
        answer_citations=[
            AnswerCitation(
                chunk_id=101,
                rank=1,
                relevance_score=0.5,
                quote="Полный текст chunk",
                act_title="Водный кодекс Российской Федерации",
                doc_number="74-ФЗ",
                doc_date="2006-06-03",
                structure_ref="Статья 6",
                article_no="6",
                clause_range="1",
            )
        ],
        retrieval_method="weighted_hybrid",
        retrieved_chunk_ids=[101, 102],
        dropped_chunk_ids=[],
        llm_model="fake",
        prompt_version="answer_v1",
        latency_ms=50,
    )


def make_answer_page() -> dict[str, object]:
    return {
        "answer": {
            "answer_id": 42,
            "question": "Что такое водные объекты общего пользования?",
            "answer_text": "Ответ по контексту.",
            "needs_clarification": False,
            "retrieval_method": "weighted_hybrid",
            "llm_model": "fake",
            "prompt_version": "answer_v1",
            "latency_ms": 50,
        },
        "citations": [
            {
                "rank": 1,
                "chunk_id": 101,
                "act_id": 7,
                "act_title": "Водный кодекс Российской Федерации",
                "doc_number": "74-ФЗ",
                "doc_date": "2006-06-03",
                "structure_ref": "Статья 6",
                "article_no": "6",
                "clause_range": "1",
                "quote": "Полный текст chunk",
            }
        ],
    }


def make_source_page() -> dict[str, object]:
    return {
        "answer_id": 42,
        "act": {
            "title": "Водный кодекс Российской Федерации",
            "doc_type": "Кодекс",
            "doc_number": "74-ФЗ",
            "doc_date": "2006-06-03",
            "edition_as_of": "2026-03-01",
            "edition_note": "Редакция актуальна для корпуса",
            "status": "actual",
        },
        "chunks": [
            {
                "chunk_id": 101,
                "chunk_index": 0,
                "text": "Полный текст chunk",
                "structure_ref": "Статья 6",
                "article_no": "6",
                "clause_range": "1",
                "token_count": 120,
                "is_cited": True,
                "citation_rank": 1,
            }
        ],
    }


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_index_returns_form_and_examples(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("RAG по федеральным нормативным актам", response.text)
        self.assertIn("Введите юридический вопрос…", response.text)
        self.assertNotIn("Frame 1", response.text)
        self.assertNotIn("LawRAG", response.text)
        self.assertNotIn("Спросите по правовому корпусу", response.text)
        self.assertNotIn("Идет поиск по корпусу и подготовка цитат.", response.text)
        self.assertIn("Что такое водные объекты общего пользования?", response.text)

    def test_post_ask_form_redirects_to_answer_page(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.generate_answer", return_value=make_generated_answer()),
            patch("law_qa_rag.web.app.save_answer_run", return_value=42),
        ):
            response = self.client.post(
                "/ask",
                data={"question": "Что такое водные объекты общего пользования?"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "http://testserver/answers/42")

    def test_post_ask_json_returns_answer_payload(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.generate_answer", return_value=make_generated_answer()),
            patch("law_qa_rag.web.app.save_answer_run", return_value=42),
        ):
            response = self.client.post(
                "/ask",
                json={"question": "Что такое водные объекты общего пользования?"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer_id"], 42)
        self.assertEqual(payload["retrieval_method"], "weighted_hybrid")
        self.assertEqual(payload["answer_citations"][0]["chunk_id"], 101)

    def test_post_ask_json_rejects_empty_question(self) -> None:
        response = self.client.post("/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Введите вопрос.")

    def test_answer_page_shows_answer_and_citation_metadata(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.load_answer_page", return_value=make_answer_page()),
        ):
            response = self.client.get("/answers/42")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Ответ по контексту.", response.text)
        self.assertIn("Водный кодекс Российской Федерации", response.text)
        self.assertIn("Статья 6", response.text)

    def test_source_page_highlights_cited_chunks(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.load_source_page", return_value=make_source_page()),
        ):
            response = self.client.get("/sources/7?answer_id=42")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Водный кодекс Российской Федерации", response.text)
        self.assertIn("цитата #1", response.text)
        self.assertIn("Полный текст chunk", response.text)


if __name__ == "__main__":
    unittest.main()
