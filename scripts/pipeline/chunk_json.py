from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase


STRUCTURAL_NODE_TYPES = {
    "part",
    "section",
    "subsection",
    "chapter",
    "paragraph_group",
    "heading",
}

TEXT_NODE_TYPES = {"paragraph", "preamble"}

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.yaml"


@dataclass(frozen=True)
class ChunkConfig:
    embedding_model: str = "BAAI/bge-m3"
    chunk_profile: str | None = None
    chunk_size_tokens: int = 800
    overlap_tokens: int = 120
    min_chunk_size_tokens: int = 100
    include_act_title: bool = True
    include_path: bool = True
    long_node_min_body_tokens: int = 50
    header_reserve_tokens: int = 80
    settings_path: str | None = None


@dataclass
class ChunkRuntime:
    config: ChunkConfig
    tokenizer: PreTrainedTokenizerBase
    token_cache: dict[tuple[str, bool], int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: ChunkConfig) -> "ChunkRuntime":
        tokenizer = AutoTokenizer.from_pretrained(config.embedding_model, use_fast=True)
        return cls(config=config, tokenizer=tokenizer)

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        """
        Подсчет токенов tokenizer'ом embedding-модели.
        Для chunk целиком считаем add_special_tokens=True, потому что именно так
        текст фактически будет подаваться в модель.
        Для отдельных абзацев можно использовать add_special_tokens=False.
        """
        cache_key = (text, add_special_tokens)
        cached = self.token_cache.get(cache_key)
        if cached is not None:
            return cached

        count = len(
            self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,
                verbose=False,
            )
        )
        self.token_cache[cache_key] = count
        return count


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_settings(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"settings file must contain a YAML mapping: {path}")
    return data


def as_positive_int(value: Any, field_name: str, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc

    if allow_zero:
        if result < 0:
            raise ValueError(f"{field_name} must be >= 0")
    elif result <= 0:
        raise ValueError(f"{field_name} must be > 0")

    return result


def as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def pick_setting(
    mappings: list[dict[str, Any]],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for mapping in mappings:
        for key in keys:
            if key in mapping:
                return mapping[key]
    return default


def get_embedding_model(settings: dict[str, Any]) -> str:
    embedding = settings.get("embedding") or {}
    if not isinstance(embedding, dict):
        return ChunkConfig.embedding_model
    return embedding.get("embedding_model") or embedding.get("model") or ChunkConfig.embedding_model


def resolve_chunk_profile(settings: dict[str, Any], requested_profile: str | None) -> str | None:
    chunking = settings.get("chunking") or {}
    if not isinstance(chunking, dict):
        return requested_profile

    if requested_profile:
        return requested_profile

    default_profile = chunking.get("default_profile")
    if isinstance(default_profile, str):
        return default_profile

    return "base" if isinstance(chunking.get("base"), dict) else None


def build_chunk_config(
    settings: dict[str, Any],
    settings_path: Path | None,
    requested_profile: str | None,
) -> ChunkConfig:
    chunking = settings.get("chunking") or {}
    if not isinstance(chunking, dict):
        chunking = {}

    profile = resolve_chunk_profile(settings, requested_profile)
    profile_values = chunking.get(profile) if profile else {}
    if profile and not isinstance(profile_values, dict):
        raise ValueError(f"chunking profile not found or invalid: {profile}")
    if not isinstance(profile_values, dict):
        profile_values = {}

    setting_layers = [profile_values, chunking]
    chunk_size = pick_setting(
        setting_layers,
        ("chunk_size_tokens", "chunk_size"),
        ChunkConfig.chunk_size_tokens,
    )
    overlap = pick_setting(
        setting_layers,
        ("overlap_tokens", "chunk_overlap_tokens", "overlap"),
        ChunkConfig.overlap_tokens,
    )
    min_chunk_size = pick_setting(
        setting_layers,
        ("min_chunk_size_tokens", "min_chunk_size"),
        ChunkConfig.min_chunk_size_tokens,
    )
    include_act_title = pick_setting(
        setting_layers,
        ("include_act_title",),
        ChunkConfig.include_act_title,
    )
    include_path = pick_setting(
        setting_layers,
        ("include_path",),
        ChunkConfig.include_path,
    )
    long_node_min_body_tokens = pick_setting(
        setting_layers,
        ("long_node_min_body_tokens",),
        ChunkConfig.long_node_min_body_tokens,
    )
    header_reserve_tokens = pick_setting(
        setting_layers,
        ("header_reserve_tokens",),
        ChunkConfig.header_reserve_tokens,
    )

    config = ChunkConfig(
        embedding_model=get_embedding_model(settings),
        chunk_profile=profile,
        chunk_size_tokens=as_positive_int(chunk_size, "chunk_size_tokens"),
        overlap_tokens=as_positive_int(overlap, "overlap_tokens", allow_zero=True),
        min_chunk_size_tokens=as_positive_int(
            min_chunk_size,
            "min_chunk_size_tokens",
            allow_zero=True,
        ),
        include_act_title=as_bool(include_act_title, "include_act_title"),
        include_path=as_bool(include_path, "include_path"),
        long_node_min_body_tokens=as_positive_int(
            long_node_min_body_tokens,
            "long_node_min_body_tokens",
        ),
        header_reserve_tokens=as_positive_int(
            header_reserve_tokens,
            "header_reserve_tokens",
            allow_zero=True,
        ),
        settings_path=str(settings_path) if settings_path and settings_path.exists() else None,
    )
    validate_chunk_config(config)
    return config


def validate_chunk_config(config: ChunkConfig) -> None:
    if config.overlap_tokens >= config.chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
    if config.min_chunk_size_tokens > config.chunk_size_tokens:
        raise ValueError("min_chunk_size_tokens must be <= chunk_size_tokens")


def node_to_text(node: dict[str, Any]) -> str:
    node_type = node.get("node_type")

    if node_type in TEXT_NODE_TYPES:
        return normalize_space(node.get("raw_text") or node.get("text") or "")

    return normalize_space(node.get("text") or "")


def build_path_lines(group: dict[str, Any], nodes: list[dict[str, Any]]) -> list[str]:
    """
    Функция строит заголовочную часть chunk.
    """
    lines: list[str] = []

    context: dict[str, Any] = {}
    for node in nodes:
        if isinstance(node.get("context"), dict):
            context = node["context"]
            break

    for key in ("part", "section", "subsection", "chapter", "paragraph_group"):
        value = context.get(key)
        if value and value not in lines:
            lines.append(value)

    article_title = group.get("article_title") or context.get("article")
    if group.get("group_type") == "article" and article_title:
        if article_title not in lines:
            lines.append(article_title)

    if group.get("group_type") == "preamble":
        if "Преамбула" not in lines:
            lines.append("Преамбула")

    return lines


def make_chunk_text(
    act: dict[str, Any],
    group: dict[str, Any],
    nodes: list[dict[str, Any]],
    config: ChunkConfig,
) -> str:
    """
    Собирает итоговый чанк из названия акта, пути, текста нормы.
    """
    parts: list[str] = []

    if config.include_act_title:
        title = normalize_space(act.get("title") or "")
        if title:
            parts.append(title)

    if config.include_path:
        path_lines = build_path_lines(group, nodes)
        if path_lines:
            parts.append("\n".join(path_lines))

    body_lines: list[str] = []
    for node in nodes:
        node_type = node.get("node_type")

        if node_type in STRUCTURAL_NODE_TYPES or node_type == "article":
            continue

        text = node_to_text(node)
        if text:
            body_lines.append(text)

    if body_lines:
        parts.append("\n".join(body_lines))

    return "\n\n".join(parts).strip()


def group_nodes_by_article(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Проход сверху вниз по нодам.

    Каждая нода article открывает новую группу. Все последующие ноды абзацев/преамбул
    попадают в эту группу до следующей article. Текст перед первой статьей
    рассматривается как преамбула.
    """
    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None

    ordered_nodes = sorted(nodes, key=lambda n: int(n.get("order", 0)))

    for node in ordered_nodes:
        node_type = node.get("node_type")

        if node_type == "article":
            if current_group is not None and current_group.get("nodes"):
                groups.append(current_group)

            current_group = {
                "group_type": "article",
                "article_no": node.get("article_no"),
                "article_title": node.get("text"),
                "structure_ref": node.get("structure_ref"),
                "nodes": [node],
            }
            continue

        if node_type in TEXT_NODE_TYPES:
            if current_group is None:
                current_group = {
                    "group_type": "preamble",
                    "article_no": None,
                    "article_title": None,
                    "structure_ref": "Преамбула",
                    "nodes": [],
                }

            current_group["nodes"].append(node)
            continue

        if node_type in STRUCTURAL_NODE_TYPES:
            continue

        continue

    if current_group is not None and current_group.get("nodes"):
        groups.append(current_group)

    return groups


def get_source_anchors(nodes: list[dict[str, Any]]) -> list[str]:
    """
    Собирает HTML-якоря исходных абзацев.
    """
    anchors: list[str] = []
    seen: set[str] = set()

    for node in nodes:
        anchor = node.get("source_anchor")
        if anchor and anchor not in seen:
            anchors.append(anchor)
            seen.add(anchor)

    return anchors


def get_node_order_range(nodes: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """
    Берёт минимальный и максимальный order среди nodes.
    """
    orders = [int(node["order"]) for node in nodes if node.get("order") is not None]
    if not orders:
        return None, None
    return min(orders), max(orders)


def get_clause_range(nodes: list[dict[str, Any]]) -> str | None:
    """
    Собирает диапазон пунктов/частей.
    Если номера идут не по возрастанию, диапазон не строится.
    """
    clauses: list[str] = []
    for node in nodes:
        clause = node.get("clause_no")
        if clause and clause not in clauses:
            clauses.append(str(clause))

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    sort_keys = [clause_sort_key(clause) for clause in clauses]
    if any(key is None for key in sort_keys):
        return None

    if sort_keys != sorted(sort_keys):
        return None

    return f"{clauses[0]}-{clauses[-1]}"


def clause_sort_key(clause: str) -> tuple[int, ...] | None:
    parts = clause.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def structure_ref_for_chunk(group: dict[str, Any], nodes: list[dict[str, Any]]) -> str | None:
    """
    Создаёт человекочитаемую ссылку на chunk.
    """
    base = group.get("structure_ref")

    if not base:
        for node in nodes:
            if node.get("structure_ref"):
                base = node["structure_ref"]
                break

    clause_range = get_clause_range(nodes)
    if base and clause_range and group.get("group_type") == "article":
        return f"{base} / п. {clause_range}"

    return base


def make_hash(
    canonical_key: str,
    structure_ref: str | None,
    text: str,
    start_node_order: int | None,
    end_node_order: int | None,
) -> str:
    """
    Создаёт SHA-256 hash chunk.

    В hash включается позиция chunk в исходном parsed JSON.
    Это нужно, потому что в юридических текстах могут быть одинаковые
    фрагменты вроде 'Статья утратила силу.'.
    """
    hash_input = (
        f"{canonical_key}\n"
        f"{structure_ref or ''}\n"
        f"{start_node_order if start_node_order is not None else ''}\n"
        f"{end_node_order if end_node_order is not None else ''}\n"
        f"{text}"
    )
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def make_chunk_record(
    act: dict[str, Any],
    group: dict[str, Any],
    nodes: list[dict[str, Any]],
    runtime: ChunkRuntime,
) -> dict[str, Any]:
    """
    Создаёт уже готовую запись chunk: объект для chunks.jsonl.
    """
    config = runtime.config
    text = make_chunk_text(act, group, nodes, config)
    structure_ref = structure_ref_for_chunk(group, nodes)
    start_order, end_order = get_node_order_range(nodes)

    record = {
        "canonical_key": act["canonical_key"],
        "chunk_index": None,
        "text": text,
        "structure_ref": structure_ref,
        "article_no": group.get("article_no"),
        "clause_range": get_clause_range(nodes),
        "source_anchors": get_source_anchors(nodes),
        "start_node_order": start_order,
        "end_node_order": end_order,
        "token_count": runtime.count_tokens(text, add_special_tokens=True),
    }

    record["hash"] = make_hash(
        canonical_key=record["canonical_key"],
        structure_ref=record["structure_ref"],
        text=record["text"],
        start_node_order=record["start_node_order"],
        end_node_order=record["end_node_order"],
    )

    return record


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    runtime: ChunkRuntime,
) -> list[str]:
    """
    Fallback для одного слишком длинного абзаца.
    Режет исходный текст по offset_mapping tokenizer'а, не декодируя токены обратно.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if overlap_tokens >= max_tokens:
        overlap_tokens = max(0, max_tokens // 4)

    encoded = runtime.tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )

    offsets = encoded.get("offset_mapping", [])
    if not offsets:
        return []

    parts: list[str] = []
    start_token = 0

    while start_token < len(offsets):
        end_token = min(start_token + max_tokens, len(offsets))

        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]

        piece = text[start_char:end_char].strip()
        if piece:
            parts.append(piece)

        if end_token >= len(offsets):
            break

        start_token = max(0, end_token - overlap_tokens)

    return parts


def split_long_node(
    node: dict[str, Any],
    max_tokens: int,
    overlap_tokens: int,
    runtime: ChunkRuntime,
) -> list[dict[str, Any]]:
    """
    Если один абзац сам длиннее chunk_size, разбивает его на виртуальные nodes.
    source_anchor/order сохраняются от исходной node.
    """
    raw_text = node_to_text(node)
    if runtime.count_tokens(raw_text, add_special_tokens=False) <= max_tokens:
        return [node]

    parts = split_text_by_tokens(
        raw_text,
        max_tokens=max_tokens,
        overlap_tokens=min(overlap_tokens, max(0, max_tokens // 4)),
        runtime=runtime,
    )

    result: list[dict[str, Any]] = []

    for i, part in enumerate(parts):
        new_node = dict(node)
        new_node["raw_text"] = part
        new_node["text"] = part
        new_node["split_part"] = i + 1
        new_node["split_total"] = len(parts)

        if i > 0:
            new_node["clause_no"] = None

        result.append(new_node)

    return result


def expand_oversized_body_nodes(
    act: dict[str, Any],
    group: dict[str, Any],
    body_nodes: list[dict[str, Any]],
    runtime: ChunkRuntime,
) -> list[dict[str, Any]]:
    """
    Функция заранее проверяет все абзацы статьи.
    Если какой-то один абзац не помещается в chunk, он разбивается через split_long_node.
    """
    config = runtime.config
    header_only_text = make_chunk_text(act, group, [], config)
    header_tokens = runtime.count_tokens(header_only_text, add_special_tokens=True)
    max_body_tokens = max(
        config.long_node_min_body_tokens,
        config.chunk_size_tokens - header_tokens - config.header_reserve_tokens,
    )

    expanded: list[dict[str, Any]] = []
    for node in body_nodes:
        candidate_text = make_chunk_text(act, group, [node], config)
        candidate_tokens = runtime.count_tokens(candidate_text, add_special_tokens=True)
        if candidate_tokens > config.chunk_size_tokens:
            expanded.extend(
                split_long_node(
                    node,
                    max_tokens=max_body_tokens,
                    overlap_tokens=config.overlap_tokens,
                    runtime=runtime,
                )
            )
        else:
            expanded.append(node)

    return expanded


def get_overlap_nodes(
    nodes: list[dict[str, Any]],
    overlap_tokens: int,
    runtime: ChunkRuntime,
) -> list[dict[str, Any]]:
    """
    Возвращает последние целые абзацы предыдущего chunk для overlap.

    Overlap считается не как фиксированное число абзацев, а как целые абзацы
    с ограничением по токенам. Например, при overlap_tokens=120 берутся последние
    абзацы предыдущего chunk, пока их суммарный размер не превышает примерно 120
    токенов. Если последний абзац сам длиннее лимита, он не переносится.
    """
    if overlap_tokens <= 0:
        return []

    selected_reversed: list[dict[str, Any]] = []
    total_tokens = 0

    for node in reversed(nodes):
        node_tokens = runtime.count_tokens(node_to_text(node), add_special_tokens=False)

        if node_tokens > overlap_tokens:
            break

        if total_tokens + node_tokens > overlap_tokens:
            break

        selected_reversed.append(node)
        total_tokens += node_tokens

    return list(reversed(selected_reversed))


def split_group_into_chunks(
    act: dict[str, Any],
    group: dict[str, Any],
    runtime: ChunkRuntime,
) -> list[dict[str, Any]]:
    """
    Функция разбиения статьи. Правило:
    - если статья помещается в chunk_size, одна статья = один chunk;
    - если она не помещается, разделить по абзацам/преамбулам;
    - overlap делается целыми абзацами с ограничением по токенам.
    """
    config = runtime.config
    group_nodes = group.get("nodes", [])
    if not group_nodes:
        return []

    full_chunk = make_chunk_record(act, group, group_nodes, runtime)
    if full_chunk["token_count"] <= config.chunk_size_tokens:
        return [full_chunk]

    body_nodes = [node for node in group_nodes if node.get("node_type") in TEXT_NODE_TYPES]

    if not body_nodes:
        return []

    body_nodes = expand_oversized_body_nodes(act, group, body_nodes, runtime)

    chunks: list[dict[str, Any]] = []
    current_nodes: list[dict[str, Any]] = []
    i = 0

    while i < len(body_nodes):
        next_node = body_nodes[i]
        candidate_nodes = current_nodes + [next_node]
        candidate = make_chunk_record(act, group, candidate_nodes, runtime)

        if candidate["token_count"] <= config.chunk_size_tokens:
            current_nodes.append(next_node)
            i += 1
            continue

        if not current_nodes:
            chunks.append(candidate)
            i += 1
            continue

        chunks.append(make_chunk_record(act, group, current_nodes, runtime))

        current_nodes = get_overlap_nodes(current_nodes, config.overlap_tokens, runtime)

        if current_nodes:
            overlap_candidate = make_chunk_record(
                act,
                group,
                current_nodes + [next_node],
                runtime,
            )
            if overlap_candidate["token_count"] > config.chunk_size_tokens:
                current_nodes = []

    if current_nodes:
        chunks.append(make_chunk_record(act, group, current_nodes, runtime))

    return chunks


def clean_act_for_jsonl(act: dict[str, Any]) -> dict[str, Any]:
    """
    Оставляет у акта только поля, которые нужны для чистой БД.
    """
    allowed = [
        "canonical_key",
        "act_kind",
        "doc_type",
        "title",
        "doc_number",
        "doc_date",
        "official_text_kind",
        "edition_as_of",
        "edition_note",
        "status",
        "has_future_editions",
        "source_file",
        "source_system",
    ]
    return {key: act.get(key) for key in allowed}


def chunk_document(
    parsed: dict[str, Any],
    runtime: ChunkRuntime,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """
    Обрабатывает один акт.
    """
    config = runtime.config
    act = parsed["act"]
    nodes = parsed.get("nodes", [])

    groups = group_nodes_by_article(nodes)

    chunks: list[dict[str, Any]] = []
    oversized_count = 0
    undersized_count = 0

    for group in groups:
        group_chunks = split_group_into_chunks(act, group, runtime)

        for chunk in group_chunks:
            chunk["chunk_index"] = len(chunks)

            if chunk["token_count"] > config.chunk_size_tokens:
                chunk["oversized"] = True
                oversized_count += 1
            else:
                chunk["oversized"] = False

            if chunk["token_count"] < config.min_chunk_size_tokens:
                chunk["undersized"] = True
                undersized_count += 1
            else:
                chunk["undersized"] = False

            chunks.append(chunk)

    stats = {
        "canonical_key": act.get("canonical_key"),
        "title": act.get("title"),
        "node_count": len(nodes),
        "group_count": len(groups),
        "chunk_count": len(chunks),
        "oversized_chunks": oversized_count,
        "undersized_chunks": undersized_count,
        "max_chunk_tokens": max((c["token_count"] for c in chunks), default=0),
    }

    return clean_act_for_jsonl(act), chunks, stats


def validate_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    """
    Проверяет уже готовые chunks.
    """
    warnings: list[str] = []
    hashes: set[str] = set()

    by_act: dict[str, list[int]] = {}

    for row_no, chunk in enumerate(chunks, start=1):
        prefix = f"chunk row {row_no}"

        if not chunk.get("canonical_key"):
            warnings.append(f"{prefix}: missing canonical_key")

        if chunk.get("chunk_index") is None:
            warnings.append(f"{prefix}: missing chunk_index")

        if not normalize_space(chunk.get("text") or ""):
            warnings.append(f"{prefix}: empty text")

        if not chunk.get("token_count") or chunk["token_count"] <= 0:
            warnings.append(f"{prefix}: invalid token_count")

        if not chunk.get("hash"):
            warnings.append(f"{prefix}: missing hash")
        elif chunk["hash"] in hashes:
            warnings.append(f"{prefix}: duplicate hash {chunk['hash']}")
        else:
            hashes.add(chunk["hash"])

        start_order = chunk.get("start_node_order")
        end_order = chunk.get("end_node_order")
        if start_order is not None and end_order is not None and end_order < start_order:
            warnings.append(f"{prefix}: end_node_order < start_node_order")

        key = chunk.get("canonical_key") or "<missing>"
        by_act.setdefault(key, []).append(chunk.get("chunk_index"))

    for key, indexes in by_act.items():
        expected = list(range(len(indexes)))
        if indexes != expected:
            warnings.append(f"{key}: chunk_index is not consecutive: {indexes[:10]}...")

    return warnings


def iter_input_files(input_dir: Path) -> list[Path]:
    files = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"manifest.json", "chunk_manifest.json"}:
            continue
        if path.name.startswith(("~$", ".")):
            continue
        files.append(path)
    return files


def process_corpus(input_dir: Path, output_dir: Path, runtime: ChunkRuntime) -> None:
    """
    Главная функция обработки всего корпуса.

    1) создаёт выходную папку;
    2) находит все parsed JSON;
    3) обрабатывает каждый акт;
    4) собирает все acts;
    5) собирает все chunks;
    6) запускает валидацию;
    7) пишет выходные файлы.
    """
    config = runtime.config
    output_dir.mkdir(parents=True, exist_ok=True)

    acts_rows: list[dict[str, Any]] = []
    chunks_rows: list[dict[str, Any]] = []
    act_stats: list[dict[str, Any]] = []

    files = iter_input_files(input_dir)
    if not files:
        raise RuntimeError(f"No parsed JSON files found in {input_dir}")

    for path in files:
        parsed = read_json(path)

        if "act" not in parsed or "nodes" not in parsed:
            print(f"[WARN] skip non-parsed file: {path.name}")
            continue

        if not parsed.get("nodes"):
            print(f"[WARN] skip parsed file without nodes: {path.name}")
            continue

        act_row, chunks, stats = chunk_document(parsed, runtime)

        acts_rows.append(act_row)
        chunks_rows.extend(chunks)
        act_stats.append(stats)

        print(
            f"[OK] {act_row.get('title') or path.name}: "
            f"{stats['node_count']} nodes -> {stats['chunk_count']} chunks"
        )

        if stats["oversized_chunks"]:
            print(
                f"[WARN] {act_row.get('canonical_key')}: "
                f"{stats['oversized_chunks']} oversized chunks"
            )

    warnings = validate_chunks(chunks_rows)
    for warning in warnings:
        print(f"[WARN] {warning}")

    acts_path = output_dir / "acts.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "chunk_manifest.json"

    write_jsonl(acts_path, acts_rows)
    write_jsonl(chunks_path, chunks_rows)

    manifest = {
        "config": {
            "settings_path": config.settings_path,
            "chunk_profile": config.chunk_profile,
            "embedding_model": config.embedding_model,
            "chunk_size_tokens": config.chunk_size_tokens,
            "overlap_tokens": config.overlap_tokens,
            "chunk_overlap_tokens": config.overlap_tokens,
            "min_chunk_size_tokens": config.min_chunk_size_tokens,
            "include_act_title": config.include_act_title,
            "include_path": config.include_path,
            "long_node_min_body_tokens": config.long_node_min_body_tokens,
            "header_reserve_tokens": config.header_reserve_tokens,
            "overlap_mode": "whole_paragraphs_limited_by_tokens",
        },
        "act_count": len(acts_rows),
        "chunk_count": len(chunks_rows),
        "avg_chunks_per_act": round(len(chunks_rows) / len(acts_rows), 2) if acts_rows else 0,
        "max_chunk_tokens": max((c["token_count"] for c in chunks_rows), default=0),
        "oversized_chunks": sum(1 for c in chunks_rows if c.get("oversized")),
        "undersized_chunks": sum(1 for c in chunks_rows if c.get("undersized")),
        "validation_warning_count": len(warnings),
        "validation_warnings": warnings[:100],
        "acts": act_stats,
    }
    write_json(manifest_path, manifest)

    print(f"[OK] wrote {acts_path}")
    print(f"[OK] wrote {chunks_path}")
    print(f"[OK] wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build article-aware chunks from parsed legal JSON files"
    )
    parser.add_argument("input_dir", type=Path, help="Directory with parsed JSON files")
    parser.add_argument("output_dir", type=Path, help="Output directory")

    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--chunk-profile",
        type=str,
        default=None,
        help="Chunking profile from settings.yaml, for example small/base/large",
    )
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--min-chunk-size", type=int, default=None)

    parser.add_argument(
        "--no-act-title",
        action="store_true",
        help="Do not prepend act title to every chunk",
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Do not prepend structural path to every chunk",
    )

    args = parser.parse_args()

    settings = read_settings(args.settings)
    config = build_chunk_config(
        settings=settings,
        settings_path=args.settings,
        requested_profile=args.chunk_profile,
    )

    overrides: dict[str, Any] = {}
    if args.embedding_model is not None:
        overrides["embedding_model"] = args.embedding_model
    if args.chunk_size is not None:
        overrides["chunk_size_tokens"] = as_positive_int(
            args.chunk_size,
            "chunk_size_tokens",
        )
    if args.chunk_overlap is not None:
        overrides["overlap_tokens"] = as_positive_int(
            args.chunk_overlap,
            "overlap_tokens",
            allow_zero=True,
        )
    if args.min_chunk_size is not None:
        overrides["min_chunk_size_tokens"] = as_positive_int(
            args.min_chunk_size,
            "min_chunk_size_tokens",
            allow_zero=True,
        )
    if args.no_act_title:
        overrides["include_act_title"] = False
    if args.no_path:
        overrides["include_path"] = False

    if overrides:
        config = replace(config, **overrides)
        validate_chunk_config(config)

    runtime = ChunkRuntime.from_config(config)
    print(
        f"[OK] config: profile={config.chunk_profile or 'default'}, "
        f"chunk_size={config.chunk_size_tokens}, overlap={config.overlap_tokens}, "
        f"model={config.embedding_model}"
    )

    process_corpus(args.input_dir, args.output_dir, runtime)


if __name__ == "__main__":
    main()
