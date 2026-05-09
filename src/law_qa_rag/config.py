from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.yaml"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Настройки embedding-модели для dense retrieval."""

    model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024


@dataclass(frozen=True)
class RetrievalConfig:
    """Настройки выбора и ранжирования chunks."""

    method: str = "weighted_hybrid"
    top_k: int = 10
    candidate_limit: int = 50
    rrf_k: int = 60
    sparse_weight: float = 0.4
    dense_weight: float = 0.6


@dataclass(frozen=True)
class LLMConfig:
    """Настройки генерации ответа."""

    provider: str = "gigachat"
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 1200
    context_token_budget: int = 6000
    prompt_version: str = "answer_v1"


@dataclass(frozen=True)
class AppConfig:
    """Собранная конфигурация RAG-пайплайна."""

    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    settings_path: Path | None = None


def read_settings(path: Path | None = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Читает YAML-настройки проекта."""
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Файл настроек должен быть YAML mapping: {path}")
    return data


def as_positive_int(value: Any, field_name: str) -> int:
    """Преобразует значение в положительное целое число."""
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} должно быть целым числом") from exc

    if result <= 0:
        raise ValueError(f"{field_name} должно быть больше 0")
    return result


def as_non_negative_float(value: Any, field_name: str) -> float:
    """Преобразует значение в неотрицательное число."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} должно быть числом") from exc

    if result < 0:
        raise ValueError(f"{field_name} должно быть не меньше 0")
    return result


def build_embedding_config(settings: dict[str, Any]) -> EmbeddingConfig:
    """Собирает настройки embeddings из YAML."""
    embedding = settings.get("embedding") or {}
    if not isinstance(embedding, dict):
        embedding = {}

    return EmbeddingConfig(
        model_name=str(embedding.get("embedding_model") or "BAAI/bge-m3"),
        embedding_dim=as_positive_int(embedding.get("embedding_dim") or 1024, "embedding_dim"),
    )


def build_retrieval_config(settings: dict[str, Any]) -> RetrievalConfig:
    """Собирает настройки retrieval из YAML."""
    retrieval = settings.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        retrieval = {}

    method = str(retrieval.get("method") or retrieval.get("hybrid_method") or "weighted_hybrid")
    if method not in {"sparse", "dense", "weighted_hybrid"}:
        raise ValueError("retrieval.method должен быть sparse, dense или weighted_hybrid")

    return RetrievalConfig(
        method=method,
        top_k=as_positive_int(retrieval.get("top_k") or 10, "retrieval.top_k"),
        candidate_limit=as_positive_int(
            retrieval.get("candidate_limit") or retrieval.get("top_k") or 50,
            "retrieval.candidate_limit",
        ),
        rrf_k=as_positive_int(retrieval.get("rrf_k") or 60, "retrieval.rrf_k"),
        sparse_weight=as_non_negative_float(
            retrieval.get("sparse_weight", 0.4),
            "retrieval.sparse_weight",
        ),
        dense_weight=as_non_negative_float(
            retrieval.get("dense_weight", 0.6),
            "retrieval.dense_weight",
        ),
    )


def build_llm_config(settings: dict[str, Any]) -> LLMConfig:
    """Собирает настройки LLM из YAML."""
    llm = settings.get("llm") or {}
    if not isinstance(llm, dict):
        llm = {}

    model = llm.get("model")
    model_name = str(model) if model not in (None, "") else None

    return LLMConfig(
        provider=str(llm.get("provider") or "gigachat"),
        model=model_name,
        temperature=float(llm.get("temperature", 0.0)),
        max_output_tokens=as_positive_int(
            llm.get("max_output_tokens") or 1200,
            "llm.max_output_tokens",
        ),
        context_token_budget=as_positive_int(
            llm.get("context_token_budget") or 6000,
            "llm.context_token_budget",
        ),
        prompt_version=str(llm.get("prompt_version") or "answer_v1"),
    )


def load_config(path: Path | None = DEFAULT_SETTINGS_PATH) -> AppConfig:
    """Загружает все настройки приложения."""
    settings = read_settings(path)
    return AppConfig(
        embedding=build_embedding_config(settings),
        retrieval=build_retrieval_config(settings),
        llm=build_llm_config(settings),
        settings_path=path if path and path.exists() else None,
    )
