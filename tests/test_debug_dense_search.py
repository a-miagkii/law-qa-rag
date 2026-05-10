from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.debug.debug_dense_search import (
    build_embedding_config,
    count_embeddings_for_model,
    embedding_to_pgvector,
    make_preview,
    positive_int,
    read_settings,
)


class FakeCursor:
    def __init__(self, row: dict[str, str]) -> None:
        self.row = row
        self.params: tuple[str, ...] | None = None

    def execute(self, query: str, params: tuple[str, ...]) -> None:
        self.params = params

    def fetchone(self) -> dict[str, str]:
        return self.row


class DebugDenseSearchTests(unittest.TestCase):
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

    def test_make_preview_truncates_long_text(self) -> None:
        self.assertEqual(make_preview("abcdef", 3), "abc\n...")
        self.assertEqual(make_preview("abc", 10), "abc")

    def test_count_embeddings_for_model_reads_dict_row(self) -> None:
        cur = FakeCursor({"with_model_embeddings": "7"})

        self.assertEqual(count_embeddings_for_model(cur, "test/model"), 7)
        self.assertEqual(cur.params, ("test/model",))


if __name__ == "__main__":
    unittest.main()
