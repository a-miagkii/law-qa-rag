from __future__ import annotations

import os
import socket
import ssl
from typing import Any
from urllib.error import URLError

from law_qa_rag.llm.base import LLMMessage, LLMResponse, TokenCount


DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 1.0


class GigaChatProvider:
    """LLMProvider поверх официального GigaChat SDK."""

    def __init__(self, model: str | None = None, timeout: float | None = None) -> None:
        self.model = model
        self.timeout = timeout

    def _client_kwargs(self) -> dict[str, Any]:
        """Собирает параметры клиента, не дублируя env-конфиг SDK."""
        kwargs: dict[str, Any] = {
            "timeout": self.timeout
            if self.timeout is not None
            else _env_float("GIGACHAT_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            "max_retries": _env_int("GIGACHAT_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            "retry_backoff_factor": _env_float(
                "GIGACHAT_RETRY_BACKOFF_FACTOR",
                DEFAULT_RETRY_BACKOFF_FACTOR,
            ),
        }
        if self.model:
            kwargs["model"] = self.model

        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if credentials:
            kwargs["credentials"] = credentials

        access_token = os.getenv("GIGACHAT_ACCESS_TOKEN")
        if access_token:
            kwargs["access_token"] = access_token

        verify_ssl = _env_bool("GIGACHAT_VERIFY_SSL_CERTS")
        if verify_ssl is not None:
            kwargs["verify_ssl_certs"] = verify_ssl

        ca_bundle = os.getenv("GIGACHAT_CA_BUNDLE_FILE")
        if ca_bundle:
            kwargs["ca_bundle_file"] = ca_bundle

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

        try:
            with self._create_client() as client:
                response = client.chat(Chat(**chat_kwargs))
        except Exception as exc:
            raise _connection_error(exc) from exc

        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        usage_dict = _usage_to_dict(usage)
        response_model = getattr(response, "model", None) or self.model
        return LLMResponse(content=content, model=response_model, usage=usage_dict)

    def count_tokens(self, texts: list[str], model: str | None = None) -> list[TokenCount]:
        """Считает токены через GigaChat tokens_count."""
        target_model = model or self.model
        kwargs = {"model": target_model} if target_model else {}

        try:
            with self._create_client() as client:
                try:
                    counts = client.tokens_count(input_=texts, **kwargs)
                except TypeError:
                    counts = client.tokens_count(texts, **kwargs)
        except Exception as exc:
            raise _connection_error(exc) from exc

        return [
            TokenCount(
                tokens=int(getattr(item, "tokens")),
                characters=int(getattr(item, "characters", len(text))),
            )
            for item, text in zip(counts, texts, strict=True)
        ]

    def list_models(self) -> list[str]:
        """Возвращает список доступных моделей GigaChat."""
        try:
            with self._create_client() as client:
                models = client.get_models()
        except Exception as exc:
            raise _connection_error(exc) from exc

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
        raise RuntimeError(f"Модель {model!r} недоступна. Доступные модели: {', '.join(available)}")


def ensure_gigachat_credentials() -> None:
    """Проверяет, что в окружении есть GigaChat credentials."""
    if not os.getenv("GIGACHAT_CREDENTIALS") and not os.getenv("GIGACHAT_ACCESS_TOKEN"):
        raise RuntimeError(
            "Нужны GigaChat credentials: задайте GIGACHAT_CREDENTIALS или GIGACHAT_ACCESS_TOKEN."
        )


def _env_bool(name: str) -> bool | None:
    """Читает boolean env-переменную."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"{name} должно быть true или false")


def _env_float(name: str, default: float) -> float:
    """Читает float env-переменную с default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} должно быть числом") from exc
    if result <= 0:
        raise RuntimeError(f"{name} должно быть больше 0")
    return result


def _env_int(name: str, default: int) -> int:
    """Читает int env-переменную с default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} должно быть целым числом") from exc
    if result < 0:
        raise RuntimeError(f"{name} должно быть не меньше 0")
    return result


def _connection_error(exc: Exception) -> RuntimeError:
    """Формирует понятную ошибку подключения к GigaChat."""
    message = str(exc)
    if "GigaChat credentials" in message:
        return RuntimeError(message)

    hint = (
        "Не удалось подключиться к GigaChat. Проверьте интернет/VPN, "
        "GIGACHAT_CREDENTIALS и доступность api.gigachat.devices.sberbank.ru. "
        "Если ошибка связана с self-signed certificate, для локальной разработки "
        "задайте GIGACHAT_VERIFY_SSL_CERTS=false или укажите GIGACHAT_CA_BUNDLE_FILE."
    )

    if _is_network_error(exc) or message:
        return RuntimeError(f"{hint} Исходная ошибка: {message}")
    return RuntimeError(hint)


def _is_network_error(exc: BaseException) -> bool:
    """Определяет сетевые и SSL-ошибки в цепочке исключений."""
    current: BaseException | None = exc
    while current is not None:
        if isinstance(
            current,
            (
                TimeoutError,
                ConnectionError,
                ConnectionResetError,
                socket.timeout,
                ssl.SSLError,
                URLError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
