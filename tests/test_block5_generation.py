from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np

from law_qa_rag.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    build_retrieval_config,
)
from law_qa_rag.generation import (
    apply_token_budget,
    build_answer_citations,
    parse_model_answer,
)
from law_qa_rag.llm.base import LLMMessage, LLMResponse, TokenCount
from law_qa_rag.llm.gigachat_client import GigaChatProvider
from law_qa_rag.prompting import build_answer_messages
from law_qa_rag.retrieval import RetrievedChunk, search_dense, weighted_rrf_fusion
from law_qa_rag import retrieval


class FakeProvider:
    def __init__(self) -> None:
        self.count_token_calls: list[list[str]] = []

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        return LLMResponse(
            content='{"answer": "Ответ", "used_chunk_ids": [1], "needs_clarification": false}',
            model="fake",
        )

    def count_tokens(self, texts: list[str], model: str | None = None) -> list[TokenCount]:
        self.count_token_calls.append(texts)
        counts = []
        for text in texts:
            tokens = 1000 if "bbbbbbbbbb" in text else 100
            counts.append(TokenCount(tokens=tokens, characters=len(text)))
        return counts

    def list_models(self) -> list[str]:
        return ["fake"]


class FakeCursor:
    def __init__(self) -> None:
        self.query: str | None = None
        self.params: dict[str, object] | None = None

    def execute(self, query: str, params: dict[str, object]) -> None:
        self.query = query
        self.params = params

    def fetchall(self) -> list[dict[str, object]]:
        return []


def make_row(chunk_id: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "chunk_id": chunk_id,
        "act_id": 10,
        "act_title": "Тестовый закон",
        "doc_number": "1-ФЗ",
        "doc_date": "2024-01-01",
        "chunk_index": chunk_id,
        "structure_ref": "Статья 1",
        "article_no": "1",
        "clause_range": None,
        "token_count": 10,
        "embedding_model": None,
        "sparse_score": None,
        "dense_score": None,
        "distance": None,
        "full_text": f"Текст chunk {chunk_id}",
    }
    row.update(overrides)
    return row


def make_chunk(chunk_id: int, text: str = "Текст нормы") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        act_id=10,
        act_title="Тестовый закон",
        doc_number="1-ФЗ",
        doc_date="2024-01-01",
        chunk_index=chunk_id,
        structure_ref="Статья 1",
        article_no="1",
        clause_range=None,
        token_count=10,
        full_text=text,
        retrieval_score=1.0 / chunk_id,
    )


class Block5GenerationTests(unittest.TestCase):
    def test_retrieval_method_is_read_from_settings(self) -> None:
        config = build_retrieval_config(
            {
                "retrieval": {
                    "method": "dense",
                    "top_k": 3,
                    "candidate_limit": 12,
                }
            }
        )

        self.assertEqual(config.method, "dense")
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.candidate_limit, 12)

    def test_weighted_hybrid_uses_weights_not_unweighted_rrf(self) -> None:
        sparse_rows = [
            make_row(1, sparse_score=0.9),
            make_row(2, sparse_score=0.8),
        ]
        dense_rows = [
            make_row(2, dense_score=0.9, distance=0.1, embedding_model="test/model"),
        ]
        config = RetrievalConfig(
            method="weighted_hybrid",
            sparse_weight=1.0,
            dense_weight=0.0,
            rrf_k=60,
        )

        fused = weighted_rrf_fusion(sparse_rows, dense_rows, config)

        self.assertEqual([chunk.chunk_id for chunk in fused[:2]], [1, 2])

    def test_dense_search_filters_by_embedding_model(self) -> None:
        cur = FakeCursor()

        search_dense(
            cur=cur,
            query_embedding=np.array([0.1, 0.2], dtype=np.float32),
            embedding_model="test/model",
            candidate_limit=5,
        )

        self.assertIn("c.embedding_model = %(embedding_model)s::text", cur.query)
        self.assertEqual(cur.params["embedding_model"], "test/model")

    def test_embedding_model_loader_is_cached_by_model_and_device(self) -> None:
        calls: list[tuple[str, str]] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, device: str) -> None:
                calls.append((model_name, device))

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":

                class FakeModule:
                    SentenceTransformer = FakeSentenceTransformer

                return FakeModule()
            return original_import(name, *args, **kwargs)

        retrieval.clear_embedding_model_cache()
        try:
            import builtins

            original = builtins.__import__
            builtins.__import__ = fake_import
            first = retrieval.load_embedding_model("test/model", "cpu")
            second = retrieval.load_embedding_model("test/model", "cpu")
            third = retrieval.load_embedding_model("test/model", "mps")
        finally:
            builtins.__import__ = original
            retrieval.clear_embedding_model_cache()

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(calls, [("test/model", "cpu"), ("test/model", "mps")])

    def test_token_budget_drops_tail_chunks(self) -> None:
        config = AppConfig(
            embedding=EmbeddingConfig(),
            retrieval=RetrievalConfig(),
            llm=LLMConfig(context_token_budget=500),
        )
        chunks = [
            make_chunk(1, "a" * 10),
            make_chunk(2, "b" * 500),
            make_chunk(3, "c" * 10),
        ]
        provider = FakeProvider()

        result = apply_token_budget("Что проверить?", chunks, config, provider)

        self.assertEqual([chunk.chunk_id for chunk in result.selected_chunks], [1])
        self.assertEqual(result.dropped_chunk_ids, [2, 3])
        self.assertEqual(len(provider.count_token_calls), 1)
        self.assertEqual(len(provider.count_token_calls[0]), 3)

    def test_gigachat_provider_reads_connection_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GIGACHAT_CREDENTIALS": "secret",
                "GIGACHAT_TIMEOUT": "150",
                "GIGACHAT_MAX_RETRIES": "4",
                "GIGACHAT_RETRY_BACKOFF_FACTOR": "2",
                "GIGACHAT_VERIFY_SSL_CERTS": "false",
            },
        ):
            kwargs = GigaChatProvider(model="test-model")._client_kwargs()

        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["credentials"], "secret")
        self.assertEqual(kwargs["timeout"], 150.0)
        self.assertEqual(kwargs["max_retries"], 4)
        self.assertEqual(kwargs["retry_backoff_factor"], 2.0)
        self.assertFalse(kwargs["verify_ssl_certs"])

    def test_prompt_contains_question_and_chunks(self) -> None:
        messages = build_answer_messages("Что проверить?", [make_chunk(7, "Важный текст")])
        payload = "\n".join(message.content for message in messages)

        self.assertIn("Что проверить?", payload)
        self.assertIn("[chunk_id: 7]", payload)
        self.assertIn("Важный текст", payload)

    def test_valid_model_json_passes_validation(self) -> None:
        answer = parse_model_answer(
            '{"answer": "Только по контексту", "used_chunk_ids": [1], "needs_clarification": false}'
        )

        self.assertEqual(answer.used_chunk_ids, [1])
        self.assertFalse(answer.needs_clarification)

    def test_invalid_model_json_fails_validation(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_answer("не json")

    def test_unknown_used_chunk_ids_fail_validation(self) -> None:
        model_answer = parse_model_answer(
            '{"answer": "Ответ", "used_chunk_ids": [99], "needs_clarification": false}'
        )

        with self.assertRaises(ValueError):
            build_answer_citations(model_answer, [make_chunk(1)])

    def test_answer_citation_quote_is_full_chunk_text(self) -> None:
        chunk = make_chunk(1, "Полный текст chunk")
        model_answer = parse_model_answer(
            '{"answer": "Ответ", "used_chunk_ids": [1], "needs_clarification": false}'
        )

        citations = build_answer_citations(model_answer, [chunk])

        self.assertEqual(citations[0].quote, "Полный текст chunk")


if __name__ == "__main__":
    unittest.main()
