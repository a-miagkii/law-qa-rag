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
        self.assertIn("Войти", response.text)
        self.assertIn("Регистрация", response.text)
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
        self.assertIn("Войдите", response.text)

    def test_register_creates_user_and_opens_session(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.create_user",
                return_value={
                    "id": 5,
                    "external_uid": "user@example.test",
                    "display_name": "User",
                },
            ) as create_user,
        ):
            response = self.client.post(
                "/register",
                data={
                    "external_uid": "USER@example.test",
                    "display_name": "User",
                    "password": "secret",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        create_user.assert_called_once()

        index = self.client.get("/")
        self.assertIn("User", index.text)
        self.assertIn("Выйти", index.text)

    def test_login_success_sets_session_and_logout_clears_it(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.authenticate_user",
                return_value={"id": 5, "external_uid": "user", "display_name": None},
            ),
        ):
            response = self.client.post(
                "/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertIn("user", self.client.get("/").text)

        logout = self.client.post("/logout", follow_redirects=False)
        self.assertEqual(logout.status_code, 303)
        self.assertIn("Войти", self.client.get("/").text)

    def test_login_failure_shows_error(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.authenticate_user", return_value=None),
        ):
            response = self.client.post(
                "/login",
                data={"external_uid": "user", "password": "bad", "next": "/"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Неверный логин или пароль", response.text)

    def test_post_ask_form_saves_query_for_logged_in_user(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.authenticate_user",
                return_value={"id": 5, "external_uid": "user", "display_name": None},
            ),
        ):
            self.client.post(
                "/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.generate_answer", return_value=make_generated_answer()),
            patch("law_qa_rag.web.app.save_answer_run", return_value=42) as save_answer_run,
        ):
            response = self.client.post(
                "/ask",
                data={"question": "Что такое водные объекты общего пользования?"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(save_answer_run.call_args.kwargs["user_id"], 5)

    def test_answer_page_prefills_existing_feedback_for_logged_in_user(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.authenticate_user",
                return_value={"id": 5, "external_uid": "user", "display_name": None},
            ),
        ):
            self.client.post(
                "/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.load_answer_page", return_value=make_answer_page()),
            patch(
                "law_qa_rag.web.app.get_feedback_for_answer_and_user",
                return_value={"rating": 4, "comment": "Хороший ответ"},
            ),
        ):
            response = self.client.get("/answers/42?feedback_saved=1")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Спасибо, оценка сохранена.", response.text)
        self.assertIn("Хороший ответ", response.text)
        self.assertIn('value="4"', response.text)

    def test_post_feedback_requires_login(self) -> None:
        response = self.client.post(
            "/answers/42/feedback",
            data={"rating": "5", "comment": "ok"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/login", response.headers["location"])

    def test_post_feedback_saves_for_logged_in_user(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.authenticate_user",
                return_value={"id": 5, "external_uid": "user", "display_name": None},
            ),
        ):
            self.client.post(
                "/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.save_feedback", return_value=99) as save_feedback,
        ):
            response = self.client.post(
                "/answers/42/feedback",
                data={"rating": "5", "comment": "Полезно"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            "http://testserver/answers/42?feedback_saved=1",
        )
        save_feedback.assert_called_once_with(
            "postgresql://test",
            42,
            5,
            5,
            "Полезно",
        )

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
