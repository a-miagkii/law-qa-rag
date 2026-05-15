from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from law_qa_rag.generation import AnswerCitation, GeneratedAnswer
from law_qa_rag.web.app import app


TEST_USER = {"id": 5, "external_uid": "user", "display_name": None}


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
            "user_id": 5,
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
                "display_quote": "Полный текст chunk",
                "doc_date_label": "03.06.2006",
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
            "doc_date_label": "03.06.2006",
            "edition_as_of": "2026-03-01",
            "edition_as_of_label": "01.03.2026",
            "edition_note": "Редакция актуальна для корпуса",
            "status": "actual",
            "status_label": "действует",
            "show_edition_note": False,
        },
        "chunks": [
            {
                "chunk_id": 101,
                "chunk_index": 0,
                "text": "Полный текст chunk",
                "display_title": "Статья 6",
                "display_text": "Полный текст нормы",
                "structure_ref": "Статья 6",
                "article_no": "6",
                "clause_range": "1",
                "token_count": 120,
                "is_cited": True,
                "citation_rank": 1,
                "citation_display_index": 1,
            }
        ],
        "source_citations": [
            {
                "chunk_id": 101,
                "display_index": 1,
                "label": "Цитата 1 — статья 6",
            }
        ],
    }


def make_history() -> list[dict[str, object]]:
    return [
        {
            "query_id": 10,
            "question": "Что такое водные объекты общего пользования?",
            "question_created_at": "2026-05-14 10:00:00",
            "answer_id": 42,
            "answer_created_at": "2026-05-14 10:00:10",
            "needs_clarification": False,
            "citation_count": 2,
        },
        {
            "query_id": 11,
            "question": "Нужен ли дополнительный контекст?",
            "question_created_at": "2026-05-14 11:00:00",
            "answer_id": 43,
            "answer_created_at": "2026-05-14 11:00:10",
            "needs_clarification": True,
            "citation_count": 0,
        },
    ]


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_index_returns_form_and_examples(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("RAG по федеральным нормативным актам", response.text)
        self.assertIn("Введите юридический вопрос…", response.text)
        self.assertIn("Войти", response.text)
        self.assertIn("Войдите, чтобы задать вопрос", response.text)
        self.assertNotIn("Главная", response.text)
        self.assertNotIn("Примеры</a>", response.text)
        self.assertNotIn("Frame 1", response.text)
        self.assertNotIn("LawRAG", response.text)
        self.assertNotIn("Спросите по правовому корпусу", response.text)
        self.assertNotIn("Идет поиск по корпусу и подготовка цитат.", response.text)
        self.assertIn("Что такое водные объекты общего пользования?", response.text)

    def test_post_ask_requires_login_for_form(self) -> None:
        response = self.client.post(
            "/ask",
            data={"question": "Что такое водные объекты общего пользования?"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("login_required=1", response.headers["location"])

    def test_post_ask_requires_login_for_json(self) -> None:
        response = self.client.post(
            "/ask",
            json={"question": "Что такое водные объекты общего пользования?"},
        )

        self.assertEqual(response.status_code, 401)

    def test_post_ask_form_redirects_to_answer_page(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.generate_answer", return_value=make_generated_answer()),
            patch("law_qa_rag.web.app.save_answer_run", return_value=42),
        ):
            self._login_test_user()
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
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.generate_answer", return_value=make_generated_answer()),
            patch("law_qa_rag.web.app.save_answer_run", return_value=42),
        ):
            self._login_test_user()
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
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
        ):
            self._login_test_user()
            response = self.client.post("/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Введите вопрос.")

    def test_answer_page_shows_answer_and_citation_metadata(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.load_answer_page", return_value=make_answer_page()),
            patch("law_qa_rag.web.app.get_feedback_for_answer_and_user", return_value=None),
        ):
            self._login_test_user()
            response = self.client.get("/answers/42")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Ответ по контексту.", response.text)
        self.assertIn("Водный кодекс Российской Федерации", response.text)
        self.assertIn("Статья 6", response.text)
        self.assertIn("Оценка ответа", response.text)
        self.assertIn("Показать полностью", response.text)
        self.assertIn("citation-quote-details", response.text)
        self.assertIn("Открыть первоисточник", response.text)
        self.assertIn("Параметры запуска", response.text)
        self.assertIn("weighted_hybrid", response.text)
        self.assertIn("answer_v1", response.text)

    def test_answer_page_requires_login(self) -> None:
        response = self.client.get("/answers/42", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertIn("login_required=1", response.headers["location"])

    def test_answer_page_forbids_other_user_answer(self) -> None:
        page = make_answer_page()
        page["answer"]["user_id"] = 99
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.load_answer_page", return_value=page),
        ):
            self._login_test_user()
            response = self.client.get("/answers/42")

        self.assertEqual(response.status_code, 403)

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
                "/auth/register",
                data={
                    "external_uid": "USER@example.test",
                    "display_name": "User",
                    "password": "secret",
                    "password_confirm": "secret",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        create_user.assert_called_once()

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.get_user_by_id",
                return_value={"id": 5, "external_uid": "user@example.test", "display_name": "User"},
            ),
        ):
            index = self.client.get("/")
        self.assertIn(">User</a>", index.text)
        self.assertNotIn("Вы вошли как", index.text)
        self.assertNotIn("Выйти", index.text)

    def test_login_success_sets_session_and_logout_clears_it(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.authenticate_user",
                return_value={"id": 5, "external_uid": "user", "display_name": None},
            ),
        ):
            response = self.client.post(
                "/auth/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
        ):
            index_text = self.client.get("/").text
            self.assertIn(">user</a>", index_text)
            self.assertNotIn("Вы вошли как", index_text)

        logout = self.client.post("/auth/logout", follow_redirects=False)
        self.assertEqual(logout.status_code, 303)
        self.assertIn("Войти", self.client.get("/").text)

    def test_login_failure_shows_error(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.authenticate_user", return_value=None),
        ):
            response = self.client.post(
                "/auth/login",
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
                "/auth/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
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
                "/auth/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
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
        self.assertIn("login_required=1", response.headers["location"])

    def test_post_feedback_saves_for_logged_in_user(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch(
                "law_qa_rag.web.app.authenticate_user",
                return_value={"id": 5, "external_uid": "user", "display_name": None},
            ),
        ):
            self.client.post(
                "/auth/login",
                data={"external_uid": "user", "password": "secret", "next": "/"},
                follow_redirects=False,
            )

        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.load_answer_page", return_value=make_answer_page()),
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
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.load_answer_page", return_value=make_answer_page()),
            patch("law_qa_rag.web.app.load_source_page", return_value=make_source_page()),
        ):
            self._login_test_user()
            response = self.client.get("/sources/7?answer_id=42")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Водный кодекс Российской Федерации", response.text)
        self.assertIn("Первоисточник", response.text)
        self.assertIn("Вернуться к ответу", response.text)
        self.assertIn("Цитата 1", response.text)
        self.assertIn("Полный текст нормы", response.text)
        self.assertIn("действует", response.text)
        self.assertNotIn("Chunk", response.text)
        self.assertNotIn("Примечание", response.text)

    def test_source_page_forbids_other_user_answer(self) -> None:
        page = make_answer_page()
        page["answer"]["user_id"] = 99
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.load_answer_page", return_value=page),
        ):
            self._login_test_user()
            response = self.client.get("/sources/7?answer_id=42")

        self.assertEqual(response.status_code, 403)

    def test_profile_requires_login(self) -> None:
        response = self.client.get("/profile", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertIn("login_required=1", response.headers["location"])

    def test_profile_shows_user_and_question_history(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch(
                "law_qa_rag.web.app.get_user_question_history", return_value=make_history()
            ) as history,
        ):
            self._login_test_user()
            response = self.client.get("/profile")

        self.assertEqual(response.status_code, 200)
        history.assert_called_once_with("postgresql://test", 5)
        self.assertIn("Профиль", response.text)
        self.assertIn("История вопросов", response.text)
        self.assertIn("Что такое водные объекты общего пользования?", response.text)
        self.assertIn("Открыть ответ", response.text)
        self.assertIn("требует уточнения", response.text)
        self.assertIn("Выйти", response.text)
        self.assertNotIn("Вы вошли как", response.text)

    def test_profile_empty_history_message(self) -> None:
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.get_user_by_id", return_value=TEST_USER),
            patch("law_qa_rag.web.app.get_user_question_history", return_value=[]),
        ):
            self._login_test_user()
            response = self.client.get("/profile")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Вы пока не задавали вопросов.", response.text)

    def _login_test_user(self, user: dict[str, object] | None = None) -> None:
        user = user or TEST_USER
        with (
            patch("law_qa_rag.web.app._get_db_url", return_value="postgresql://test"),
            patch("law_qa_rag.web.app.authenticate_user", return_value=user),
        ):
            self.client.post(
                "/auth/login",
                data={"external_uid": user["external_uid"], "password": "secret", "next": "/"},
                follow_redirects=False,
            )


if __name__ == "__main__":
    unittest.main()
