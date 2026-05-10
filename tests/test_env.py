from __future__ import annotations

import unittest
from unittest.mock import patch

from law_qa_rag.env import build_postgres_url, get_database_url


class EnvTests(unittest.TestCase):
    def test_build_postgres_url_quotes_credentials(self) -> None:
        url = build_postgres_url(
            db_name="rag laws",
            user="rag user",
            password="pass/word",
            host="localhost",
            port="5433",
        )

        self.assertEqual(
            url,
            "postgresql://rag%20user:pass%2Fword@localhost:5433/rag%20laws",
        )

    def test_get_database_url_prefers_explicit_database_url(self) -> None:
        with (
            patch("law_qa_rag.env.load_project_dotenv", return_value=False),
            patch.dict("os.environ", {"DATABASE_URL": "postgresql://direct"}, clear=True),
        ):
            self.assertEqual(get_database_url(), "postgresql://direct")

    def test_get_database_url_builds_from_postgres_env(self) -> None:
        env = {
            "POSTGRES_DB": "rag_laws",
            "POSTGRES_USER": "rag_user",
            "POSTGRES_PASSWORD": "secret",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "5434",
        }

        with (
            patch("law_qa_rag.env.load_project_dotenv", return_value=False),
            patch.dict("os.environ", env, clear=True),
        ):
            self.assertEqual(
                get_database_url(),
                "postgresql://rag_user:secret@127.0.0.1:5434/rag_laws",
            )

    def test_get_database_url_returns_none_when_optional_and_missing(self) -> None:
        with (
            patch("law_qa_rag.env.load_project_dotenv", return_value=False),
            patch.dict("os.environ", {}, clear=True),
        ):
            self.assertIsNone(get_database_url(required=False))


if __name__ == "__main__":
    unittest.main()
