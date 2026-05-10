from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from scripts.pipeline.parse_docs import (
    collect_input_files,
    parse_act_metadata,
)


class ParseDocsTests(unittest.TestCase):
    def test_collect_input_files_skips_temporary_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.doc"
            bad = root / "~$bad.doc"
            good.write_text("<html></html>", encoding="utf-8")
            bad.write_text("<html></html>", encoding="utf-8")

            files, skipped = collect_input_files(root)

        self.assertEqual([path.name for path in files], ["good.doc"])
        self.assertEqual([path.name for path in skipped], ["~$bad.doc"])

    def test_metadata_is_detected_by_content_not_position(self) -> None:
        html = """
        <html>
          <body>
            <div id="topText">
              <div>Служебная строка</div>
              <div>Кодекс Российской Федерации от 03.06.2006 № 74-ФЗ</div>
              <div>Официальный текст</div>
              <div>Редакция с 01.03.2026, актуальная</div>
            </div>
            <p class="Z">Водный кодекс Российской Федерации</p>
          </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")

        act = parse_act_metadata(soup, Path("source.doc"))

        self.assertEqual(act["doc_date"], "2006-06-03")
        self.assertEqual(act["doc_number"], "74-ФЗ")
        self.assertEqual(act["title"], "Водный кодекс Российской Федерации")
        self.assertEqual(act["official_text_kind"], "Официальный текст")
        self.assertEqual(act["edition_as_of"], "2026-03-01")


if __name__ == "__main__":
    unittest.main()
