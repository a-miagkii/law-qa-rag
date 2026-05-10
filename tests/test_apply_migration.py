from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.db.apply_migration import read_migration_sql


class ApplyMigrationTests(unittest.TestCase):
    def test_read_migration_sql_reads_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "001_test.sql"
            path.write_text("SELECT 1;", encoding="utf-8")

            self.assertEqual(read_migration_sql(path), "SELECT 1;")

    def test_read_migration_sql_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_migration_sql(Path(tmp) / "missing.sql")


if __name__ == "__main__":
    unittest.main()
