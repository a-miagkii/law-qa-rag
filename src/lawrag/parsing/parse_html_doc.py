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


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def remove_editorial_spans(tag: Tag) -> Tag:
    cloned = normalize_superscripts(tag)
    for bad in cloned.select("span.mark, span.markx"):
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
    top = soup.find("div", id="topText")
    top_lines: list[str] = []
    if top:
        top_lines = [
            normalize_space(div.get_text(" ", strip=True))
            for div in top.find_all("div")
            if normalize_space(div.get_text(" ", strip=True))
        ]
    header = parse_header_line(top_lines[0] if len(top_lines) >= 1 else None)
    edition = parse_edition_line(top_lines[2] if len(top_lines) >= 3 else None)
    title = extract_title(soup)
    act_kind = infer_act_kind(header.get("doc_type"), title)
    act: dict[str, Any] = {
        "canonical_key": make_canonical_key(act_kind, header.get("doc_date"), header.get("doc_number")),
        "act_kind": act_kind,
        "doc_type": header.get("doc_type"),
        "title": title,
        "doc_number": header.get("doc_number"),
        "doc_date": header.get("doc_date"),
        "official_text_kind": top_lines[1] if len(top_lines) >= 2 else None,
        "edition_as_of": edition["edition_as_of"],
        "edition_note": edition["edition_note"],
        "status": edition["status"],
        "has_future_editions": edition["has_future_editions"],
        "source_file": path.name,
    }
    if header.get("header_line"):
        act["header_line"] = header["header_line"]
    return act


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
    current_context: dict[str, str | None] = {
        "part": None,
        "section": None,
        "subsection": None,
        "chapter": None,
        "paragraph_group": None,
        "article": None,
    }
    order = 0
    for p in soup.find_all("p"):
        classes = p.get("class") or []
        cls = classes[0] if classes else ""
        pid = p.get("id")
        raw_text = normalize_space(p.get_text(" ", strip=True))
        if not raw_text or raw_text == "РОССИЙСКАЯ ФЕДЕРАЦИЯ":
            continue
        text = get_clean_text(p)
        if not text:
            continue
        if cls == TITLE_CLASS or cls in SKIP_PARAGRAPH_CLASSES:
            continue
        if cls == HEADING_CLASS:
            node_type, heading_text = parse_heading_text(text)
            if node_type == "part":
                current_context.update({"part": heading_text, "section": None, "subsection": None, "chapter": None, "paragraph_group": None, "article": None})
            elif node_type == "section":
                current_context.update({"section": heading_text, "subsection": None, "chapter": None, "paragraph_group": None, "article": None})
            elif node_type == "subsection":
                current_context.update({"subsection": heading_text, "chapter": None, "paragraph_group": None, "article": None})
            elif node_type == "chapter":
                current_context.update({"chapter": heading_text, "paragraph_group": None, "article": None})
            elif node_type == "paragraph_group":
                current_context.update({"paragraph_group": heading_text, "article": None})
            elif node_type == "article":
                current_context["article"] = heading_text
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
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.doc"))
    manifest: list[dict[str, Any]] = []
    for path in files:
        parsed = parse_document(path)
        out_path = output_dir / f"{path.stem}.json"
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        act = parsed["act"]
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
        })
        print(f"[OK] {path.name} -> {out_path.name} ({len(parsed['nodes'])} nodes)")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] manifest -> {manifest_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse pravo.gov.ru HTML-exported .doc legal acts into structured JSON")
    parser.add_argument("input_path", type=Path, help="Path to .doc file or directory with .doc files")
    parser.add_argument("output_dir", type=Path, help="Output directory for parsed JSON files")
    args = parser.parse_args()
    parse_path(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
