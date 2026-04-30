from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

SKIP_PARAGRAPH_CLASSES = {"A", "C", "T"}
HEADING_CLASS = "H"
TITLE_CLASS = "Z"
SOURCE_SYSTEM = "pravo.gov.ru_html_doc"
TEMP_FILE_PREFIXES = ("~$", ".")
STRUCTURE_KEYS = (
    "part",
    "section",
    "subsection",
    "chapter",
    "paragraph_group",
    "article",
)

HEADER_RE = re.compile(
    r"^(?P<doc_type>.+?)\s+от\s+(?P<doc_date>\d{2}\.\d{2}\.\d{4})\s*№\s*(?P<doc_number>.+?)$"
)
ARTICLE_RE = re.compile(r"^Статья\s+(?P<num>\d+(?:\.\d+)?)\.?")


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "cp1251", "windows-1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode file: {path}")


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def should_skip_input_file(path: Path) -> bool:
    """
    Skips transient files produced by office editors and non-.doc inputs.
    """
    if path.suffix.lower() != ".doc":
        return True
    return path.name.startswith(TEMP_FILE_PREFIXES)


def collect_input_files(input_path: Path) -> tuple[list[Path], list[Path]]:
    candidates = [input_path] if input_path.is_file() else sorted(input_path.glob("*.doc"))
    files: list[Path] = []
    skipped: list[Path] = []

    for path in candidates:
        if should_skip_input_file(path):
            skipped.append(path)
            continue
        files.append(path)

    return files, skipped


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    # 7 .1 -> 7.1
    text = re.sub(r"(?<=\d)\s+\.\s*(?=\d)", ".", text)

    # седьмой .1 -> седьмой.1
    text = re.sub(r"(?<=[А-Яа-яA-Za-z])\s+\.\s*(?=\d)", ".", text)

    # 25.4 . -> 25.4.
    text = re.sub(r"(?<=\d)\s+\.", ".", text)

    # слово . -> слово.
    text = re.sub(r"(?<=[А-Яа-яA-Za-z])\s+\.", ".", text)

    # 12.1 ) -> 12.1)
    text = re.sub(r"(?<=\d)\s+\)", ")", text)

    # пробел перед запятой/точкой с запятой/двоеточием
    text = re.sub(r"\s+([,;:])", r"\1", text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()


def make_empty_context() -> dict[str, str | None]:
    return {key: None for key in STRUCTURE_KEYS}


def update_context_for_heading(
    ctx: dict[str, str | None],
    node_type: str,
    heading_text: str,
) -> None:
    resets_by_type: dict[str, tuple[str, ...]] = {
        "part": ("part", "section", "subsection", "chapter", "paragraph_group", "article"),
        "section": ("section", "subsection", "chapter", "paragraph_group", "article"),
        "subsection": ("subsection", "chapter", "paragraph_group", "article"),
        "chapter": ("chapter", "paragraph_group", "article"),
        "paragraph_group": ("paragraph_group", "article"),
        "article": ("article",),
    }

    keys_to_reset = resets_by_type.get(node_type)
    if not keys_to_reset:
        return

    for key in keys_to_reset:
        ctx[key] = None
    ctx[node_type] = heading_text


def to_iso_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", date_str)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    return f"{yyyy}-{mm}-{dd}"


def clone_tag(tag: Tag) -> Tag:
    soup = BeautifulSoup(str(tag), "html.parser")
    cloned = soup.find(tag.name)
    if cloned is None:
        raise ValueError("Failed to clone tag")
    return cloned


def normalize_superscripts(tag: Tag) -> Tag:
    cloned = clone_tag(tag)
    for sup in cloned.select("span.W9, span.WB"):
        sup_text = normalize_space(sup.get_text("", strip=True))
        if re.fullmatch(r"\d+", sup_text):
            sup.replace_with("." + sup_text)
    return cloned


LOST_FORCE_RE = re.compile(
    r"(утратил[ао]?|утратили|утративш\w+)\s+сил",
    flags=re.IGNORECASE,
)


def has_lost_force_marker(text: str) -> bool:
    return bool(LOST_FORCE_RE.search(normalize_space(text)))


def normalize_lost_force_marker(text: str) -> str:
    """
    Сохраняем юридически важный факт утраты силы,
    но убираем длинную ссылку на закон-изменение.
    """
    lower = normalize_space(text).lower()

    if "статья" in lower:
        return "Утратила силу."
    if "часть" in lower:
        return "Часть утратила силу."
    if "пункт" in lower:
        return "Пункт утратил силу."
    if "подпункт" in lower:
        return "Подпункт утратил силу."
    if "абзац" in lower:
        return "Абзац утратил силу."

    return "Утратила силу."


def remove_editorial_spans(tag: Tag) -> Tag:
    cloned = normalize_superscripts(tag)

    for bad in cloned.select("span.mark, span.markx"):
        marker_text = normalize_space(bad.get_text(" ", strip=True))

        if has_lost_force_marker(marker_text):
            bad.replace_with(" " + normalize_lost_force_marker(marker_text) + " ")
        else:
            bad.decompose()

    return cloned


def get_clean_text(tag: Tag) -> str:
    cleaned = remove_editorial_spans(tag)
    return normalize_space(cleaned.get_text(" ", strip=True))


def infer_act_kind(doc_type: str | None, title: str | None) -> str:
    s = normalize_space(f"{doc_type or ''} {title or ''}").lower()
    if "федеральный конституционный закон" in s:
        return "federal_constitutional_law"
    if "кодекс" in s:
        return "codex"
    if "федеральный закон" in s:
        return "federal_law"
    return "other"


def make_canonical_key(act_kind: str | None, doc_date: str | None, doc_number: str | None) -> str:
    kind = act_kind or "other"
    date = doc_date or "unknown-date"
    number = (doc_number or "unknown-number").replace(" ", "")
    return f"{kind}:{date}:{number}"


def parse_header_line(header_line: str | None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "header_line": normalize_space(header_line or "") or None,
        "doc_type": None,
        "doc_date": None,
        "doc_number": None,
    }
    if not header_line:
        return data
    m = HEADER_RE.match(normalize_space(header_line))
    if not m:
        return data
    data["doc_type"] = normalize_space(m.group("doc_type"))
    data["doc_date"] = to_iso_date(m.group("doc_date"))
    data["doc_number"] = normalize_space(m.group("doc_number"))
    return data


def parse_edition_line(edition_line: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "edition_note": normalize_space(edition_line or "") or None,
        "edition_as_of": None,
        "status": "unknown",
        "has_future_editions": False,
    }
    if not edition_line:
        return result
    line = normalize_space(edition_line)
    m = re.search(r"Редакция\s+с\s+(\d{2}\.\d{2}\.\d{4})", line, flags=re.IGNORECASE)
    if m:
        result["edition_as_of"] = to_iso_date(m.group(1))
    lower = line.lower()
    has_actual = "актуаль" in lower
    has_future = "не вступивш" in lower and "в силу" in lower
    result["has_future_editions"] = has_future
    if "утратил силу" in lower:
        result["status"] = "inactive"
    elif has_actual and has_future:
        result["status"] = "actual_with_future_editions"
    elif has_actual:
        result["status"] = "actual"
    return result


def extract_top_lines(soup: BeautifulSoup) -> list[str]:
    top = soup.find("div", id="topText")
    if not top:
        return []

    return [
        normalize_space(div.get_text(" ", strip=True))
        for div in top.find_all("div")
        if normalize_space(div.get_text(" ", strip=True))
    ]


def find_header_line(top_lines: list[str]) -> str | None:
    for line in top_lines:
        if HEADER_RE.match(normalize_space(line)):
            return line
    return top_lines[0] if top_lines else None


def find_edition_line(top_lines: list[str]) -> str | None:
    for line in top_lines:
        lower = normalize_space(line).lower()
        if "редакц" in lower or "актуаль" in lower or "утратил силу" in lower:
            return line
    return None


def find_official_text_kind(top_lines: list[str]) -> str | None:
    for line in top_lines:
        if "официальный" in normalize_space(line).lower():
            return normalize_space(line)
    return None


def extract_title(soup: BeautifulSoup) -> str | None:
    title_tag = soup.find("p", class_=TITLE_CLASS)
    if title_tag:
        title = normalize_space(title_tag.get_text(" ", strip=True))
        if title:
            return title
    for p in soup.find_all("p"):
        text = get_clean_text(p)
        if not text:
            continue
        cls = (p.get("class") or [""])[0]
        if cls in {"Z", "C", "T"} and len(text) > 8 and "РОССИЙСКАЯ ФЕДЕРАЦИЯ" not in text:
            return text
    return None


def parse_act_metadata(soup: BeautifulSoup, path: Path) -> dict[str, Any]:
    top_lines = extract_top_lines(soup)
    header = parse_header_line(find_header_line(top_lines))
    edition = parse_edition_line(find_edition_line(top_lines))
    title = extract_title(soup)
    act_kind = infer_act_kind(header.get("doc_type"), title)
    act: dict[str, Any] = {
        "canonical_key": make_canonical_key(act_kind, header.get("doc_date"), header.get("doc_number")),
        "act_kind": act_kind,
        "doc_type": header.get("doc_type"),
        "title": title,
        "doc_number": header.get("doc_number"),
        "doc_date": header.get("doc_date"),
        "official_text_kind": find_official_text_kind(top_lines),
        "edition_as_of": edition["edition_as_of"],
        "edition_note": edition["edition_note"],
        "status": edition["status"],
        "has_future_editions": edition["has_future_editions"],
        "source_file": path.name,
        "source_system": SOURCE_SYSTEM,
    }
    if header.get("header_line"):
        act["header_line"] = header["header_line"]
    return act


def validate_parsed_document(parsed: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    act = parsed.get("act") or {}
    nodes = parsed.get("nodes") or []

    if not act.get("title"):
        warnings.append("missing act title")
    if not act.get("doc_date") or not act.get("doc_number"):
        warnings.append("missing act date or number")
    if not nodes:
        warnings.append("no legal nodes parsed")

    return warnings


def parse_heading_text(text: str) -> tuple[str, str]:
    text = normalize_space(text)
    if text.startswith("Часть"):
        return "part", text
    if text.startswith("Раздел"):
        return "section", text
    if text.startswith("Подраздел"):
        return "subsection", text
    if text.startswith("Глава"):
        return "chapter", text
    if text.startswith("Параграф") or text.startswith("§"):
        return "paragraph_group", text
    if text.startswith("Статья"):
        return "article", text
    return "heading", text


def extract_article_no(article_heading: str | None) -> str | None:
    if not article_heading:
        return None
    m = ARTICLE_RE.match(normalize_space(article_heading))
    return m.group("num") if m else None


def extract_clause_info(text: str) -> tuple[str | None, str]:
    text = normalize_space(text)
    m = re.match(r"^(\d+(?:\.\d+)?)\.\s+(.*)$", text)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(\d+(?:\.\d+)?)\)\s+(.*)$", text)
    if m:
        return m.group(1), m.group(2)
    return None, text


def build_structure_ref(ctx: dict[str, str | None], clause_no: str | None = None) -> str | None:
    parts: list[str] = []
    for key in ("part", "section", "subsection", "chapter", "paragraph_group", "article"):
        value = ctx.get(key)
        if value:
            parts.append(value)
    if clause_no:
        parts.append(f"п. {clause_no}")
    return " / ".join(parts) if parts else None


def is_preamble_context(ctx: dict[str, str | None]) -> bool:
    return not ctx.get("article")


def parse_document(path: Path) -> dict[str, Any]:
    html = read_text_file(path)
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, Any] = {"act": parse_act_metadata(soup, path), "nodes": []}
    current_context = make_empty_context()
    order = 0
    for p in soup.find_all("p"):
        classes = p.get("class") or []
        cls = classes[0] if classes else ""
        pid = p.get("id")
        raw_text = normalize_space(p.get_text(" ", strip=True))
        has_legal_status = has_lost_force_marker(raw_text)
        if not raw_text or raw_text == "РОССИЙСКАЯ ФЕДЕРАЦИЯ":
            continue
        text = get_clean_text(p)
        if not text:
            continue
        if cls == TITLE_CLASS:
            continue

        if cls in SKIP_PARAGRAPH_CLASSES and not has_legal_status:
            continue

        if cls == HEADING_CLASS and has_lost_force_marker(text) and not ARTICLE_RE.match(text):
            clause_no, body_text = extract_clause_info(text)
            node_type = "preamble" if is_preamble_context(current_context) else "paragraph"

            result["nodes"].append({
                "order": order,
                "node_type": node_type,
                "text": body_text,
                "raw_text": text,
                "source_anchor": pid,
                "article_no": extract_article_no(current_context.get("article")),
                "clause_no": clause_no,
                "structure_ref": build_structure_ref(current_context, clause_no),
                "context": deepcopy(current_context),
            })
            order += 1
            continue

        if cls == HEADING_CLASS:
            node_type, heading_text = parse_heading_text(text)
            update_context_for_heading(current_context, node_type, heading_text)
            result["nodes"].append({
                "order": order,
                "node_type": node_type,
                "text": heading_text,
                "source_anchor": pid,
                "article_no": extract_article_no(current_context.get("article")),
                "clause_no": None,
                "structure_ref": build_structure_ref(current_context),
                "context": deepcopy(current_context),
            })
            order += 1
            continue
        clause_no, body_text = extract_clause_info(text)
        node_type = "preamble" if is_preamble_context(current_context) else "paragraph"
        result["nodes"].append({
            "order": order,
            "node_type": node_type,
            "text": body_text,
            "raw_text": text,
            "source_anchor": pid,
            "article_no": extract_article_no(current_context.get("article")),
            "clause_no": clause_no,
            "structure_ref": build_structure_ref(current_context, clause_no),
            "context": deepcopy(current_context),
        })
        order += 1
    return result


def parse_path(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files, skipped = collect_input_files(input_path)
    for path in skipped:
        print(f"[SKIP] {path.name}")
    if not files:
        raise RuntimeError(f"No .doc files found in {input_path}")

    manifest: list[dict[str, Any]] = []
    for path in files:
        parsed = parse_document(path)
        out_path = output_dir / f"{path.stem}.json"
        write_json(out_path, parsed)
        act = parsed["act"]
        warnings = validate_parsed_document(parsed)
        manifest.append({
            "source_file": act.get("source_file"),
            "json_file": out_path.name,
            "canonical_key": act.get("canonical_key"),
            "act_kind": act.get("act_kind"),
            "doc_type": act.get("doc_type"),
            "title": act.get("title"),
            "doc_number": act.get("doc_number"),
            "doc_date": act.get("doc_date"),
            "edition_as_of": act.get("edition_as_of"),
            "status": act.get("status"),
            "node_count": len(parsed["nodes"]),
            "warnings": warnings,
        })
        print(f"[OK] {path.name} -> {out_path.name} ({len(parsed['nodes'])} nodes)")
        for warning in warnings:
            print(f"[WARN] {path.name}: {warning}")
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[OK] manifest -> {manifest_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse pravo.gov.ru HTML-exported .doc legal acts into structured JSON")
    parser.add_argument("input_path", type=Path, help="Path to .doc file or directory with .doc files")
    parser.add_argument("output_dir", type=Path, help="Output directory for parsed JSON files")
    args = parser.parse_args()
    parse_path(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
