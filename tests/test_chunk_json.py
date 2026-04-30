from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.chunk_json import (
    build_chunk_config,
    clause_sort_key,
    iter_input_files,
)


class ChunkJsonTests(unittest.TestCase):
    def test_config_uses_requested_profile(self) -> None:
        settings = {
            "embedding": {"embedding_model": "test/model"},
            "chunking": {
                "default_profile": "base",
                "min_chunk_size_tokens": 10,
                "small": {"chunk_size_tokens": 500, "overlap_tokens": 80},
                "base": {"chunk_size_tokens": 800, "overlap_tokens": 120},
            },
        }

        config = build_chunk_config(settings, Path("settings.yaml"), "small")

        self.assertEqual(config.embedding_model, "test/model")
        self.assertEqual(config.chunk_profile, "small")
        self.assertEqual(config.chunk_size_tokens, 500)
        self.assertEqual(config.overlap_tokens, 80)
        self.assertEqual(config.min_chunk_size_tokens, 10)

    def test_clause_sort_key_keeps_dotted_integer_order(self) -> None:
        values = ["12.2", "12.10", "13"]

        self.assertEqual(sorted(values, key=clause_sort_key), values)
        self.assertEqual(clause_sort_key("12.10"), (12, 10))

    def test_iter_input_files_skips_manifests_and_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "chunk_manifest.json").write_text("{}", encoding="utf-8")
            (root / "~$bad.json").write_text("{}", encoding="utf-8")
            (root / "good.json").write_text("{}", encoding="utf-8")

            files = iter_input_files(root)

        self.assertEqual([path.name for path in files], ["good.json"])


if __name__ == "__main__":
    unittest.main()
