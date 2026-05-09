from __future__ import annotations

import os
from typing import Any

from law_qa_rag.llm.base import LLMMessage, LLMResponse, TokenCount


class GigaChatProvider:
    """LLMProvider поверх официального GigaChat SDK."""

    def __init__(self, model: str | None = None, timeout: float = 60.0) -> None:
        self.model = model
        self.timeout = timeout

    def _client_kwargs(self) -> dict[str, Any]:
        """Собирает параметры клиента, не дублируя env-конфиг SDK."""
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.model:
            kwargs["model"] = self.model
        return kwargs

    def _create_client(self) -> Any:
        """Создает GigaChat client с ленивым импортом SDK."""
        try:
            from gigachat import GigaChat
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Не установлен пакет gigachat. Выполните: pip install -r requirements.txt"
            ) from exc

        ensure_gigachat_credentials()

        return GigaChat(**self._client_kwargs())

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Генерирует chat-completion ответ через GigaChat."""
        try:
            from gigachat.models import Chat, Messages, MessagesRole
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Не установлен пакет gigachat. Выполните: pip install -r requirements.txt"
            ) from exc

        role_map = {
            "system": MessagesRole.SYSTEM,
            "user": MessagesRole.USER,
            "assistant": MessagesRole.ASSISTANT,
        }
        chat_messages = [
            Messages(role=role_map.get(message.role, MessagesRole.USER), content=message.content)
            for message in messages
        ]
        chat_kwargs: dict[str, Any] = {
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.model:
            chat_kwargs["model"] = self.model

        with self._create_client() as client:
            response = client.chat(Chat(**chat_kwargs))

        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        usage_dict = _usage_to_dict(usage)
        response_model = getattr(response, "model", None) or self.model
        return LLMResponse(content=content, model=response_model, usage=usage_dict)

    def count_tokens(self, texts: list[str], model: str | None = None) -> list[TokenCount]:
        """Считает токены через GigaChat tokens_count."""
        target_model = model or self.model
        kwargs = {"model": target_model} if target_model else {}

        with self._create_client() as client:
            try:
                counts = client.tokens_count(input_=texts, **kwargs)
            except TypeError:
                counts = client.tokens_count(texts, **kwargs)

        return [
            TokenCount(
                tokens=int(getattr(item, "tokens")),
                characters=int(getattr(item, "characters", len(text))),
            )
            for item, text in zip(counts, texts, strict=True)
        ]

    def list_models(self) -> list[str]:
        """Возвращает список доступных моделей GigaChat."""
        with self._create_client() as client:
            models = client.get_models()

        return [
            str(getattr(item, "id_", None) or getattr(item, "id", ""))
            for item in getattr(models, "data", [])
            if getattr(item, "id_", None) or getattr(item, "id", None)
        ]


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Преобразует usage SDK в обычный dict."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {str(key): int(value) for key, value in usage.items() if value is not None}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif hasattr(usage, "dict"):
        data = usage.dict()
    else:
        data = {
            key: getattr(usage, key)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "precached_prompt_tokens",
            )
            if hasattr(usage, key)
        }
    return {str(key): int(value) for key, value in data.items() if value is not None}


def validate_model_available(provider: GigaChatProvider, model: str | None) -> None:
    """Проверяет, что явно заданная модель доступна аккаунту."""
    if model is None:
        return

    available = provider.list_models()
    if model not in available:
        raise RuntimeError(
            f"Модель {model!r} недоступна. Доступные модели: {', '.join(available)}"
        )


def ensure_gigachat_credentials() -> None:
    """Проверяет, что в окружении есть GigaChat credentials."""
    if not os.getenv("GIGACHAT_CREDENTIALS") and not os.getenv("GIGACHAT_ACCESS_TOKEN"):
        raise RuntimeError(
            "Нужны GigaChat credentials: задайте GIGACHAT_CREDENTIALS "
            "или GIGACHAT_ACCESS_TOKEN."
        )
