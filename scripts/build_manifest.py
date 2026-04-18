from __future__ import annotations

import csv
import sys
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET


def strip_ns(tag: str) -> str:
    """Remove XML namespace if present."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_first(root: ET.Element, tag_name: str) -> ET.Element | None:
    """Find first element anywhere in tree by local tag name."""
    for elem in root.iter():
        if strip_ns(elem.tag) == tag_name:
            return elem
    return None


def get_attr_val(root: ET.Element, tag_name: str, attr: str = "val") -> str:
    elem = find_first(root, tag_name)
    if elem is None:
        return ""
    return (elem.attrib.get(attr) or "").strip()


def get_text(root: ET.Element, tag_name: str) -> str:
    elem = find_first(root, tag_name)
    if elem is None:
        return ""
    text = "".join(elem.itertext()).strip()
    return " ".join(text.split())


def parse_date_ru(date_str: str) -> str:
    """Convert dd.mm.yyyy -> yyyy-mm-dd. Return empty string on failure."""
    date_str = (date_str or "").strip()
    if not date_str:
        return ""
    try:
        return datetime.strptime(date_str, "%d.%m.%Y").date().isoformat()
    except ValueError:
        return ""


def parse_keywords_count(root: ET.Element) -> int:
    count = 0
    for elem in root.iter():
        if strip_ns(elem.tag) == "keywordByIPS":
            count += 1
    return count


def parse_xml_file(xml_path: Path) -> dict:
    row = {
        "file_path": str(xml_path),
        "source_nd": "",
        "doc_type": "",
        "title": "",
        "issued_by": "",
        "doc_author_normal_form": "",
        "doc_number": "",
        "doc_date_raw": "",
        "doc_date_iso": "",
        "status": "",
        "actual_datetime_human": "",
        "is_widely_used": "",
        "classifier": "",
        "keywords_count": 0,
        "text_len_chars": 0,
        "has_text": 0,
        "parse_error": "",
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        row["source_nd"] = get_attr_val(root, "pravogovruNd")
        row["doc_type"] = get_attr_val(root, "doc_typeIPS")
        row["title"] = get_text(root, "headingIPS")
        row["issued_by"] = get_attr_val(root, "issuedByIPS")
        row["doc_author_normal_form"] = get_attr_val(root, "doc_author_normal_formIPS")
        row["doc_number"] = get_attr_val(root, "docNumberIPS")
        row["doc_date_raw"] = get_attr_val(root, "docdateIPS")
        row["doc_date_iso"] = parse_date_ru(row["doc_date_raw"])
        row["status"] = get_attr_val(root, "statusIPS")
        row["actual_datetime_human"] = get_attr_val(root, "actual_datetime_humanIPS")
        row["is_widely_used"] = get_attr_val(root, "is_widely_used")
        row["classifier"] = get_attr_val(root, "classifierByIPS")
        row["keywords_count"] = parse_keywords_count(root)

        text_ips = get_text(root, "textIPS")
        row["text_len_chars"] = len(text_ips)
        row["has_text"] = 1 if text_ips else 0

    except Exception as e:
        row["parse_error"] = f"{type(e).__name__}: {e}"

    return row


def build_manifest(input_dir: Path, output_csv: Path) -> None:
    xml_files = sorted(input_dir.rglob("*.xml"))

    if not xml_files:
        print(f"No XML files found in: {input_dir}")
        return

    rows = []
    total = len(xml_files)

    for i, xml_path in enumerate(xml_files, start=1):
        if i % 1000 == 0 or i == total:
            print(f"[{i}/{total}] {xml_path}")
        rows.append(parse_xml_file(xml_path))

    fieldnames = [
        "file_path",
        "source_nd",
        "doc_type",
        "title",
        "issued_by",
        "doc_author_normal_form",
        "doc_number",
        "doc_date_raw",
        "doc_date_iso",
        "status",
        "actual_datetime_human",
        "is_widely_used",
        "classifier",
        "keywords_count",
        "text_len_chars",
        "has_text",
        "parse_error",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_manifest.py <input_dir> <output_csv>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    build_manifest(input_dir, output_csv)