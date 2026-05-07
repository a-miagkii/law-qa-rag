from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.embed_chunks import (
    build_embedding_config,
    estimate_eta_seconds,
    format_duration,
    read_settings,
)


class EmbedChunksTests(unittest.TestCase):
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

    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(12.34), "12.3с")
        self.assertEqual(format_duration(125), "2м 05с")
        self.assertEqual(format_duration(3661), "1ч 01м")

    def test_estimate_eta_seconds(self) -> None:
        self.assertEqual(estimate_eta_seconds(10, 100, 5.0), 45.0)
        self.assertIsNone(estimate_eta_seconds(0, 100, 5.0))


if __name__ == "__main__":
    unittest.main()
