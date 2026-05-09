from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMMessage:
    """Сообщение для chat-completion модели."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Ответ LLM provider."""

    content: str
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenCount:
    """Результат подсчета токенов для одного текста."""

    tokens: int
    characters: int


class LLMProvider(Protocol):
    """Минимальный интерфейс LLM provider для RAG-пайплайна."""

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Генерирует ответ по chat messages."""

    def count_tokens(self, texts: list[str], model: str | None = None) -> list[TokenCount]:
        """Считает токены для списка текстов."""

    def list_models(self) -> list[str]:
        """Возвращает доступные модели provider."""
