from __future__ import annotations

import argparse
import unittest

from scripts.debug_sparse_search import positive_int


class DebugSparseSearchTests(unittest.TestCase):
    def test_positive_int_accepts_positive_value(self) -> None:
        self.assertEqual(positive_int("10"), 10)

    def test_positive_int_rejects_negative_value(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("-1")


if __name__ == "__main__":
    unittest.main()
