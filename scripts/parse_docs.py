from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

# Служебные классы, которые не должны попадать в clean text как нормы.
# Важно: C/T больше не пропускаются безусловно, потому что в Конституции
# и некоторых HTML-выгрузках именно в C/T лежат главы и статьи.
SKIP_PARAGRAPH_CLASSES = {"A", "S"}
CENTERED_CLASSES = {"C", "T"}
HEADING_CLASS = "H"
TITLE_CLASS = "Z"

SOURCE_SYSTEM_DEFAULT = "pravo.gov.ru_html_doc"
TEMP_FILE_PREFIXES = ("~$", ".")
SUPPORTED_SUFFIXES = {".doc", ".html", ".htm"}

STRUCTURE_KEYS = (
    "part",
    "section",
    "subsection",
    "chapter",
    "paragraph_group",
    "article",
)

HEADER_RE = re.compile(
    r"^(?P<doc_type>.+?)\s+от\s+"
    r"(?P<doc_date>\d{2}\.\d{2}\.\d{4})"
    r"(?:\s*г\.?)?"
    r"\s*(?:(?:№|N)\s*)?"
    r"(?P<doc_number>б/н|[0-9]+(?:-[0-9А-Яа-яA-Za-z]+)?(?:/[0-9]+)?(?:-[0-9А-Яа-яA-Za-z]+)?)",
    flags=re.IGNORECASE,
)

ARTICLE_RE = re.compile(r"^Статья\s+(?P<num>\d+(?:\.\d+)?)\.?", flags=re.IGNORECASE)


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "windows-1251"):
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
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return True
    return path.name.startswith(TEMP_FILE_PREFIXES)


def collect_input_files(input_path: Path) -> tuple[list[Path], list[Path]]:
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(
            [
                path
                for path in input_path.iterdir()
                if path.is_file()
            ]
        )

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

    # а.1 ) -> а.1)
    text = re.sub(r"(?<=[А-Яа-яA-Za-z]\.\d)\s+\)", ")", text)

    # пробел перед запятой/точкой с запятой/двоеточием
    text = re.sub(r"\s+([,;:])", r"\1", text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()


def normalize_title_case(text: str) -> str:
    text = normalize_space(text)
    if text.upper() == "КОНСТИТУЦИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ":
        return "Конституция Российской Федерации"
    return text


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

    if "конституция российской федерации" in s:
        return "constitution"

    if "федеральный конституционный закон" in s:
        return "federal_constitutional_law"

    if "кодекс" in s:
        return "codex"

    if "федеральный закон" in s:
        return "federal_law"

    if (
        "закон российской федерации" in s
        or "закон рф" in s
        or "закон рсфср" in s
        or "закон российской советской федеративной социалистической республики" in s
    ):
        return "law_rf"

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

    line = normalize_space(header_line)
    m = HEADER_RE.match(line)
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
    if not m:
        m = re.search(r"По\s+с[оo]стоянию\s+на\s+(\d{2}\.\d{2}\.\d{4})", line, flags=re.IGNORECASE)

    if m:
        result["edition_as_of"] = to_iso_date(m.group(1))

    lower = line.lower()
    has_actual = "актуаль" in lower or "по состоянию на" in lower
    has_future = "не вступивш" in lower and "в силу" in lower

    result["has_future_editions"] = has_future

    if "утратил силу" in lower:
        result["status"] = "inactive"
    elif has_actual and has_future:
        result["status"] = "actual_with_future_editions"
    elif has_actual:
        result["status"] = "actual"

    return result


def detect_source_system(soup: BeautifulSoup) -> str:
    generator = soup.find("meta", attrs={"http-equiv": re.compile("^GENERATOR$", re.IGNORECASE)})
    if generator and "kodeks" in normalize_space(generator.get("content") or "").lower():
        return "kodeks_html"

    if soup.find("div", id="topText"):
        return "pravo.gov.ru_html_doc"

    return SOURCE_SYSTEM_DEFAULT


def extract_fallback_lines(soup: BeautifulSoup) -> list[str]:

    lines: list[str] = []

    body = soup.find("body") or soup
    for tag in body.find_all(["span", "div", "p"], limit=250):
        text = normalize_space(tag.get_text(" ", strip=True))
        if not text:
            continue

        lower = text.lower()
        is_header_like = bool(HEADER_RE.match(text))
        is_edition_like = "по состоянию на" in lower or "редакц" in lower or "актуаль" in lower

        if is_header_like or is_edition_like:
            if text not in lines:
                lines.append(text)

    return lines


def extract_top_lines(soup: BeautifulSoup) -> list[str]:
    top = soup.find("div", id="topText")
    if top:
        return [
            normalize_space(div.get_text(" ", strip=True))
            for div in top.find_all("div")
            if normalize_space(div.get_text(" ", strip=True))
        ]

    return extract_fallback_lines(soup)


def find_header_line(top_lines: list[str]) -> str | None:
    for line in top_lines:
        if HEADER_RE.match(normalize_space(line)):
            return line
    return top_lines[0] if top_lines else None


def find_edition_line(top_lines: list[str]) -> str | None:
    for line in top_lines:
        lower = normalize_space(line).lower()
        if (
            "редакц" in lower
            or "актуаль" in lower
            or "утратил силу" in lower
            or "по состоянию на" in lower
        ):
            return line
    return None


def find_official_text_kind(top_lines: list[str]) -> str | None:
    for line in top_lines:
        if "официальный" in normalize_space(line).lower():
            return normalize_space(line)
    return None


def is_bad_title_candidate(text: str) -> bool:
    lower = normalize_space(text).lower()
    if not lower:
        return True

    bad_starts = (
        "принят",
        "одобрен",
        "по состоянию",
        "редакция",
        "глава ",
        "раздел ",
        "подраздел ",
        "параграф ",
        "статья ",
    )
    if lower.startswith(bad_starts):
        return True

    bad_exact = {
        "российская федерация",
        "раздел первый",
        "раздел второй",
    }
    return lower in bad_exact


def extract_title(soup: BeautifulSoup) -> str | None:
    title_tag = soup.find("p", class_=TITLE_CLASS)
    if title_tag:
        title = normalize_title_case(title_tag.get_text(" ", strip=True))
        if title:
            return title

    html_title = soup.find("title")
    if html_title:
        title = normalize_title_case(html_title.get_text(" ", strip=True))
        if title and title.lower() not in {"complex", "document"} and not is_bad_title_candidate(title):
            return title

    for p in soup.find_all("p"):
        text = normalize_title_case(get_clean_text(p))
        if not text:
            continue

        cls = (p.get("class") or [""])[0]
        if cls in {"Z", "C", "T"} and len(text) > 8 and not is_bad_title_candidate(text):
            return text

    return None


def parse_act_metadata(soup: BeautifulSoup, path: Path) -> dict[str, Any]:
    top_lines = extract_top_lines(soup)
    header = parse_header_line(find_header_line(top_lines))
    edition = parse_edition_line(find_edition_line(top_lines))

    if not edition.get("edition_as_of"):
        full_text = normalize_space(soup.get_text(" ", strip=True))
        m = re.search(
            r"по\s+состоянию\s+на\s+(\d{2}\.\d{2}\.\d{4})",
            full_text,
            flags=re.IGNORECASE,
        )
        if m:
            edition["edition_as_of"] = to_iso_date(m.group(1))
            edition["edition_note"] = f"По состоянию на {m.group(1)}"
            edition["status"] = "actual"

    title = extract_title(soup)

    if title and "конституция российской федерации" in title.lower():
        if not header.get("doc_type"):
            header["doc_type"] = "Конституция Российской Федерации"
        if not header.get("doc_date"):
            header["doc_date"] = "1993-12-12"
        if not header.get("doc_number"):
            header["doc_number"] = "б/н"

        if not edition.get("edition_as_of"):
            edition["edition_as_of"] = "2026-05-06"
            edition["edition_note"] = "По состоянию на 06.05.2026"
            edition["status"] = "actual"

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
        "source_system": detect_source_system(soup),
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
    if not act.get("edition_as_of"):
        warnings.append("missing edition_as_of")
    if not nodes:
        warnings.append("no legal nodes parsed")

    return warnings


def parse_heading_text(text: str) -> tuple[str, str]:
    text = normalize_space(text)
    lower = text.lower()

    if lower.startswith("часть"):
        return "part", text
    if re.match(r"^раздел\b", lower):
        return "section", text
    if re.match(r"^подраздел\b", lower):
        return "subsection", text
    if re.match(r"^глава\b", lower):
        return "chapter", text
    if lower.startswith("параграф") or text.startswith("§"):
        return "paragraph_group", text
    if ARTICLE_RE.match(text):
        return "article", text
    return "heading", text


def extract_article_no(article_heading: str | None) -> str | None:
    if not article_heading:
        return None
    m = ARTICLE_RE.match(normalize_space(article_heading))
    return m.group("num") if m else None


def extract_clause_info(text: str) -> tuple[str | None, str]:
    text = normalize_space(text)

    # 1. Текст...
    m = re.match(r"^(\d+(?:\.\d+)?)\.\s+(.*)$", text)
    if m:
        return m.group(1), m.group(2)

    # 1) Текст...
    m = re.match(r"^(\d+(?:\.\d+)?)\)\s+(.*)$", text)
    if m:
        return m.group(1), m.group(2)

    # а) Текст... / а.1) Текст...
    m = re.match(r"^([А-Яа-я](?:\.\d+)?)\)\s+(.*)$", text)
    if m:
        return m.group(1), m.group(2)

    return None, text


def build_structure_ref(ctx: dict[str, str | None], clause_no: str | None = None) -> str | None:
    parts: list[str] = []
    for key in STRUCTURE_KEYS:
        value = ctx.get(key)
        if value:
            parts.append(value)
    if clause_no:
        parts.append(f"п. {clause_no}")
    return " / ".join(parts) if parts else None


def is_preamble_context(ctx: dict[str, str | None]) -> bool:
    return not ctx.get("article")


def is_document_title_text(text: str, act: dict[str, Any]) -> bool:
    title = normalize_space(act.get("title") or "")
    if not title:
        return False
    return normalize_space(text).lower() == title.lower()


def is_heading_candidate(cls: str, text: str, act_kind: str | None) -> bool:
    node_type, _ = parse_heading_text(text)

    if cls == HEADING_CLASS:
        return True

    if node_type in {"part", "section", "subsection", "chapter", "paragraph_group", "article"}:
        return True

    if act_kind == "constitution" and cls in CENTERED_CLASSES and node_type != "heading":
        return True

    return False


def is_constitution_chapter_title(
    cls: str,
    text: str,
    act: dict[str, Any],
    ctx: dict[str, str | None],
) -> bool:
    """
    In Constitution HTML chapter number and chapter name are separate paragraphs:

    <p class="C">ГЛАВА 2</p>
    <p class="T">ПРАВА И СВОБОДЫ ЧЕЛОВЕКА И ГРАЖДАНИНА</p>

    This function identifies the second line so we can merge it into:
    'ГЛАВА 2. ПРАВА И СВОБОДЫ ЧЕЛОВЕКА И ГРАЖДАНИНА'
    """
    if act.get("act_kind") != "constitution":
        return False

    if cls not in CENTERED_CLASSES:
        return False

    if not ctx.get("chapter") or ctx.get("article"):
        return False

    if is_document_title_text(text, act):
        return False

    node_type, _ = parse_heading_text(text)
    if node_type != "heading":
        return False

    letters = re.findall(r"[А-Яа-яA-Za-z]", text)
    if not letters:
        return False

    upper_letters = re.findall(r"[А-ЯA-Z]", text)
    upper_ratio = len(upper_letters) / max(1, len(letters))

    return upper_ratio > 0.75 and len(text) <= 180


def merge_constitution_chapter_title(
    current_context: dict[str, str | None],
    nodes: list[dict[str, Any]],
    title_text: str,
) -> None:
    chapter = current_context.get("chapter")
    if not chapter:
        return

    if title_text in chapter:
        return

    combined = f"{chapter}. {title_text}"
    current_context["chapter"] = combined


    for node in reversed(nodes):
        if node.get("node_type") == "chapter":
            node["text"] = combined
            node["structure_ref"] = build_structure_ref(current_context)
            node["context"] = deepcopy(current_context)
            break


def should_skip_service_paragraph(
    cls: str,
    text: str,
    act_kind: str | None,
    has_legal_status: bool,
) -> bool:
    if has_legal_status:
        return False

    if cls in SKIP_PARAGRAPH_CLASSES:
        return True

    if act_kind != "constitution" and cls in CENTERED_CLASSES:
        return True

    return False




def is_signature_start(cls: str, text: str) -> bool:
    """
    Detects final presidential signature block.
    In pravo.gov.ru/KODEKS exports it is usually p.Y:
    'Президент Российской Федерации ...'.
    This is metadata/signature, not normative text, so it should not be attached
    to the last article as a paragraph.
    """
    normalized = normalize_space(text)

    if cls == "Y" and normalized.startswith("Президент Российской Федерации"):
        return True

    return bool(
        re.fullmatch(
            r"Президент Российской Федерации\s+[А-ЯA-Z]\.\s*[А-Яа-яA-Za-z-]+",
            normalized,
        )
    )


def make_source_anchor(tag: Tag, fallback_order: int) -> str | None:
    return (
        tag.get("id")
        or tag.get("data-n")
        or tag.get("name")
        or f"p{fallback_order}"
    )


def append_heading_node(
    result: dict[str, Any],
    order: int,
    node_type: str,
    heading_text: str,
    source_anchor: str | None,
    current_context: dict[str, str | None],
) -> None:
    result["nodes"].append({
        "order": order,
        "node_type": node_type,
        "text": heading_text,
        "source_anchor": source_anchor,
        "article_no": extract_article_no(current_context.get("article")),
        "clause_no": None,
        "structure_ref": build_structure_ref(current_context),
        "context": deepcopy(current_context),
    })


def append_text_node(
    result: dict[str, Any],
    order: int,
    text: str,
    source_anchor: str | None,
    current_context: dict[str, str | None],
) -> None:
    clause_no, body_text = extract_clause_info(text)
    node_type = "preamble" if is_preamble_context(current_context) else "paragraph"

    result["nodes"].append({
        "order": order,
        "node_type": node_type,
        "text": body_text,
        "raw_text": text,
        "source_anchor": source_anchor,
        "article_no": extract_article_no(current_context.get("article")),
        "clause_no": clause_no,
        "structure_ref": build_structure_ref(current_context, clause_no),
        "context": deepcopy(current_context),
    })


def parse_document(path: Path) -> dict[str, Any]:
    html = read_text_file(path)
    soup = BeautifulSoup(html, "lxml")

    result: dict[str, Any] = {"act": parse_act_metadata(soup, path), "nodes": []}
    current_context = make_empty_context()
    order = 0
    final_signature_started = False

    act_kind = result["act"].get("act_kind")

    for p_index, p in enumerate(soup.find_all("p")):
        classes = p.get("class") or []
        cls = classes[0] if classes else ""
        pid = make_source_anchor(p, p_index)

        raw_text = normalize_space(p.get_text(" ", strip=True))
        has_legal_status = has_lost_force_marker(raw_text)

        if not raw_text or raw_text == "РОССИЙСКАЯ ФЕДЕРАЦИЯ":
            continue

        text = get_clean_text(p)
        if not text:
            continue

        if is_document_title_text(text, result["act"]):
            continue

        if cls == TITLE_CLASS:
            continue

        if final_signature_started:
            continue

        if is_signature_start(cls, text):
            final_signature_started = True
            continue

        if is_constitution_chapter_title(cls, text, result["act"], current_context):
            merge_constitution_chapter_title(current_context, result["nodes"], text)
            continue

        if is_heading_candidate(cls, text, act_kind):
            if cls == HEADING_CLASS and has_lost_force_marker(text) and not ARTICLE_RE.match(text):
                append_text_node(
                    result=result,
                    order=order,
                    text=text,
                    source_anchor=pid,
                    current_context=current_context,
                )
                order += 1
                continue

            node_type, heading_text = parse_heading_text(text)

            update_context_for_heading(current_context, node_type, heading_text)

            append_heading_node(
                result=result,
                order=order,
                node_type=node_type,
                heading_text=heading_text,
                source_anchor=pid,
                current_context=current_context,
            )
            order += 1
            continue

        if should_skip_service_paragraph(cls, text, act_kind, has_legal_status):
            continue

        append_text_node(
            result=result,
            order=order,
            text=text,
            source_anchor=pid,
            current_context=current_context,
        )
        order += 1

    return result


def parse_path(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files, skipped = collect_input_files(input_path)

    for path in skipped:
        print(f"[SKIP] {path.name}")

    if not files:
        raise RuntimeError(
            f"No supported files found in {input_path}. "
            f"Supported suffixes: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

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
            "source_system": act.get("source_system"),
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
    parser = argparse.ArgumentParser(
        description="Parse HTML-exported legal acts into structured JSON"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to .doc/.html file or directory with supported files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for parsed JSON files",
    )
    args = parser.parse_args()

    parse_path(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
