from __future__ import annotations

from law_qa_rag.llm.base import LLMMessage
from law_qa_rag.retrieval import RetrievedChunk


ANSWER_PROMPT_VERSION = "answer_v1"


SYSTEM_PROMPT_V1 = """
Ты юридический ассистент RAG-системы. Отвечай только по переданному контексту.
Не используй внешние знания и не додумывай отсутствующие факты.

Верни только валидный JSON. Не используй markdown. Не добавляй текст до или после JSON.
JSON должен начинаться с { и заканчиваться }.

Поле answer должно быть одной строкой без переносов строк.
Не используй двойные кавычки внутри answer; при необходимости используй кавычки «ёлочки».
Ответ должен быть кратким, но достаточным: обычно 4–8 предложений.
Если вопрос требует нескольких норм, условий или исключений, допускается до 10 предложений.
Не пересказывай весь фрагмент закона целиком. Укажи правовой вывод, условия, сроки и важные исключения.

Если в контексте недостаточно информации для ответа, верни:
{
  "answer": "В переданном контексте недостаточно информации для ответа на вопрос.",
  "used_chunk_ids": [],
  "needs_clarification": true
}

Если ответ можно дать по контексту, верни JSON строго такого вида:
{
  "answer": "строка с ответом на русском языке",
  "used_chunk_ids": [123, 456],
  "needs_clarification": false
}

В used_chunk_ids включай только chunk_id из контекста, на которые реально опирается ответ.
Если used_chunk_ids пустой, needs_clarification должен быть true.
""".strip()


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Собирает текстовый контекст из chunks."""
    parts = []
    for chunk in chunks:
        source = (
            f"{chunk.act_title}"
            f" от {chunk.doc_date or 'дата не указана'}"
            f" № {chunk.doc_number or 'номер не указан'}"
        )
        parts.append(
            "\n".join(
                [
                    f"[chunk_id: {chunk.chunk_id}]",
                    f"Источник: {source}",
                    f"Структура: {chunk.structure_ref or 'не указана'}",
                    f"Статья: {chunk.article_no or 'не указана'}",
                    f"Пункты: {chunk.clause_range or 'не указаны'}",
                    "Текст:",
                    chunk.full_text.strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def build_answer_messages(
    question: str,
    chunks: list[RetrievedChunk],
    prompt_version: str = ANSWER_PROMPT_VERSION,
) -> list[LLMMessage]:
    """Собирает messages для генерации ответа."""
    if prompt_version != ANSWER_PROMPT_VERSION:
        raise ValueError(f"Неизвестная версия prompt: {prompt_version}")

    user_prompt = "\n\n".join(
        [
            f"Вопрос: {question.strip()}",
            "Контекст:",
            build_context(chunks) if chunks else "Контекст отсутствует.",
            "Ответь строго в JSON по заданной схеме.",
        ]
    )
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT_V1),
        LLMMessage(role="user", content=user_prompt),
    ]


def serialize_messages(messages: list[LLMMessage]) -> str:
    """Сериализует messages для подсчета токенов."""
    return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages)
