from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from scripts.debug.debug_hybrid_search import (
    build_embedding_config,
    count_embeddings_for_model,
    embedding_to_pgvector,
    encode_query,
    positive_int,
    read_settings,
    rrf_fusion,
    search_dense,
)


class FakeCursor:
    def __init__(self, row: dict[str, str] | None = None) -> None:
        self.row = row or {}
        self.query: str | None = None
        self.params: Any = None

    def execute(self, query: str, params: Any = None) -> None:
        self.query = query
        self.params = params

    def fetchone(self) -> dict[str, str]:
        return self.row

    def fetchall(self) -> list[dict[str, Any]]:
        return []


class FakeModel:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding

    def encode(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return self.embedding


def make_row(chunk_id: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "chunk_id": chunk_id,
        "act_id": 1,
        "act_title": "Тестовый акт",
        "doc_number": "1-ФЗ",
        "doc_date": "2024-01-01",
        "chunk_index": chunk_id,
        "structure_ref": "Статья 1",
        "article_no": "1",
        "clause_range": None,
        "token_count": 10,
        "embedding_model": None,
        "full_text": "Текст chunk",
        "sparse_score": None,
        "dense_score": None,
        "distance": None,
    }
    row.update(overrides)
    return row


class DebugHybridSearchTests(unittest.TestCase):
    def test_positive_int_accepts_positive_value(self) -> None:
        self.assertEqual(positive_int("10"), 10)

    def test_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_config_uses_settings_embedding_values(self) -> None:
        settings = {
            "embedding": {
                "embedding_model": "test/model",
                "embedding_dim": 384,
            },
        }

        config = build_embedding_config(settings, Path("settings.yaml"))

        self.assertEqual(config.model_name, "test/model")
        self.assertEqual(config.embedding_dim, 384)

    def test_read_settings_returns_empty_mapping_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.yaml"

            self.assertEqual(read_settings(path), {})

    def test_embedding_to_pgvector_formats_dense_vector(self) -> None:
        value = np.array([0.1, -0.25, 1.0], dtype=np.float32)

        self.assertEqual(
            embedding_to_pgvector(value),
            "[0.10000000,-0.25000000,1.00000000]",
        )

    def test_encode_query_checks_embedding_dim(self) -> None:
        model = FakeModel(np.array([[0.1, 0.2]], dtype=np.float32))

        result = encode_query(model, "запрос", expected_dim=2)

        np.testing.assert_array_equal(
            result,
            np.array([0.1, 0.2], dtype=np.float32),
        )

    def test_encode_query_rejects_unexpected_dim(self) -> None:
        model = FakeModel(np.array([[0.1, 0.2]], dtype=np.float32))

        with self.assertRaises(RuntimeError):
            encode_query(model, "запрос", expected_dim=3)

    def test_count_embeddings_for_model_reads_dict_row(self) -> None:
        cur = FakeCursor({"with_model_embeddings": "7"})

        self.assertEqual(count_embeddings_for_model(cur, "test/model"), 7)
        self.assertEqual(cur.params, ("test/model",))

    def test_search_dense_filters_by_embedding_model(self) -> None:
        cur = FakeCursor()

        search_dense(
            cur=cur,
            query_embedding=np.array([0.1, 0.2], dtype=np.float32),
            model_name="test/model",
            candidate_limit=5,
            act_filter=None,
            article_no=None,
        )

        self.assertIn("c.embedding_model = %(embedding_model)s::text", cur.query)
        self.assertEqual(cur.params["embedding_model"], "test/model")

    def test_rrf_fusion_boosts_rows_seen_by_both_retrievers(self) -> None:
        sparse_results = [
            make_row(1, sparse_score=0.9),
            make_row(2, sparse_score=0.8),
        ]
        dense_results = [
            make_row(2, dense_score=0.7, distance=0.3, embedding_model="test/model"),
            make_row(3, dense_score=0.6, distance=0.4, embedding_model="test/model"),
        ]

        fused = rrf_fusion(sparse_results, dense_results, rrf_k=60)

        self.assertEqual([row["chunk_id"] for row in fused], [2, 1, 3])
        self.assertEqual(fused[0]["sparse_rank"], 2)
        self.assertEqual(fused[0]["dense_rank"], 1)
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])


if __name__ == "__main__":
    unittest.main()
