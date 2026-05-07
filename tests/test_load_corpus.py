from __future__ import annotations

import argparse
import unittest

from scripts.load_corpus import positive_int


class LoadCorpusTests(unittest.TestCase):
    def test_positive_int_accepts_positive_value(self) -> None:
        self.assertEqual(positive_int("1000"), 1000)

    def test_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")


if __name__ == "__main__":
    unittest.main()
