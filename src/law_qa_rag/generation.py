from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ValidationError

from law_qa_rag.config import AppConfig
from law_qa_rag.llm.base import LLMProvider, LLMResponse
from law_qa_rag.prompting import build_answer_messages, serialize_messages
from law_qa_rag.retrieval import RetrievedChunk, retrieve_chunks


class ModelAnswer(BaseModel):
    """Структурированный JSON, который должна вернуть модель."""

    answer: str
    used_chunk_ids: list[int]
    needs_clarification: bool


@dataclass(frozen=True)
class TokenBudgetResult:
    """Итог применения token budget к контексту."""

    selected_chunks: list[RetrievedChunk]
    dropped_chunk_ids: list[int]
    total_tokens: int


@dataclass(frozen=True)
class AnswerCitation:
    """Детерминированная цитата по chunk_id."""

    chunk_id: int
    rank: int
    relevance_score: float
    quote: str
    act_title: str
    doc_number: str | None
    doc_date: str | None
    structure_ref: str | None
    article_no: str | None
    clause_range: str | None


@dataclass(frozen=True)
class GeneratedAnswer:
    """Полный JSON-результат generation pipeline."""

    answer: str
    used_chunk_ids: list[int]
    needs_clarification: bool
    answer_citations: list[AnswerCitation]
    retrieval_method: str
    retrieved_chunk_ids: list[int]
    dropped_chunk_ids: list[int]
    llm_model: str | None
    prompt_version: str
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Преобразует результат в JSON-serializable dict."""
        data = asdict(self)
        data["answer_citations"] = [asdict(citation) for citation in self.answer_citations]
        return data


def apply_token_budget(
    question: str,
    chunks: list[RetrievedChunk],
    config: AppConfig,
    provider: LLMProvider,
) -> TokenBudgetResult:
    """Оставляет top chunks, которые помещаются в token budget."""
    if not chunks:
        messages = build_answer_messages(question, [], config.llm.prompt_version)
        total_tokens = provider.count_tokens(
            [serialize_messages(messages)],
            model=config.llm.model,
        )[0].tokens
        return TokenBudgetResult(
            selected_chunks=[],
            dropped_chunk_ids=[],
            total_tokens=total_tokens,
        )

    candidate_payloads = [
        serialize_messages(
            build_answer_messages(
                question,
                chunks[: index + 1],
                config.llm.prompt_version,
            )
        )
        for index in range(len(chunks))
    ]
    token_counts = provider.count_tokens(candidate_payloads, model=config.llm.model)

    selected_count = 0
    total_tokens = 0
    for index, token_count in enumerate(token_counts):
        if token_count.tokens <= config.llm.context_token_budget:
            selected_count = index + 1
            total_tokens = token_count.tokens
        else:
            break

    selected = chunks[:selected_count]
    dropped = [item.chunk_id for item in chunks[selected_count:]]
    if selected_count == 0:
        total_tokens = token_counts[0].tokens

    return TokenBudgetResult(
        selected_chunks=selected,
        dropped_chunk_ids=dropped,
        total_tokens=total_tokens,
    )


def parse_model_answer(raw_content: str) -> ModelAnswer:
    """Парсит и валидирует JSON модели.

    Парсер намеренно допускает только безопасные исправления:
    - удаление markdown-обертки;
    - извлечение JSON-объекта из ответа;
    - добавление недостающей закрывающей фигурной скобки, если JSON явно оборвался в конце;
    - разбор JSON со строгой последующей валидацией через ModelAnswer.

    Обычный текст вместо JSON не преобразуется в ответ.
    Если модель не указала used_chunk_ids, содержательный ответ не принимается:
    система возвращает безопасное сообщение о недостаточности контекста.
    """
    content = _strip_json_fence(raw_content).strip()

    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    def finalize_answer(answer: ModelAnswer) -> ModelAnswer:
        if not isinstance(answer.answer, str) or not answer.answer.strip():
            raise ValueError("Поле answer должно быть непустой строкой")

        if not answer.used_chunk_ids:
            safe_update = {
                "answer": "В переданном контексте недостаточно информации для ответа на вопрос.",
                "used_chunk_ids": [],
                "needs_clarification": True,
            }

            if hasattr(answer, "model_copy"):
                return answer.model_copy(update=safe_update)

            return answer.copy(update=safe_update)

        return answer

    add_candidate(content)

    json_start = content.find("{")
    json_end = content.rfind("}")

    if json_start >= 0 and json_end > json_start:
        add_candidate(content[json_start : json_end + 1])

    if json_start >= 0:
        maybe_json = content[json_start:].strip()
        open_braces = maybe_json.count("{")
        close_braces = maybe_json.count("}")
        missing_braces = open_braces - close_braces

        if 0 < missing_braces <= 3:
            add_candidate(maybe_json + ("}" * missing_braces))

    last_exc: Exception | None = None

    for candidate in candidates:
        try:
            if hasattr(ModelAnswer, "model_validate_json"):
                answer = ModelAnswer.model_validate_json(candidate)
            else:
                answer = ModelAnswer.parse_raw(candidate)

            return finalize_answer(answer)

        except (ValidationError, ValueError, TypeError) as exc:
            last_exc = exc

        try:
            data = json.loads(candidate, strict=False)

            if hasattr(ModelAnswer, "model_validate"):
                answer = ModelAnswer.model_validate(data)
            else:
                answer = ModelAnswer.parse_obj(data)

            return finalize_answer(answer)

        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            last_exc = exc

    debug_dir = Path(os.getenv("RAG_LLM_DEBUG_DIR", "logs/llm_debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)

    debug_path = debug_dir / f"invalid_llm_response_{time.time_ns()}.txt"
    debug_path.write_text(content, encoding="utf-8")

    last_debug_path = debug_dir / "last_invalid_llm_response.txt"
    last_debug_path.parent.mkdir(parents=True, exist_ok=True)
    last_debug_path.write_text(content, encoding="utf-8")

    raise ValueError(
        f"Модель вернула невалидный JSON: {last_exc}. Сырой ответ сохранен в {debug_path}"
    ) from last_exc


def build_answer_citations(
    model_answer: ModelAnswer,
    selected_chunks: list[RetrievedChunk],
) -> list[AnswerCitation]:
    """Собирает точные цитаты по used_chunk_ids."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in selected_chunks}
    unknown_ids = [
        chunk_id for chunk_id in model_answer.used_chunk_ids if chunk_id not in chunks_by_id
    ]
    if unknown_ids:
        raise ValueError(f"Модель сослалась на неизвестные chunk_id: {unknown_ids}")

    citations = []
    seen: set[int] = set()
    for rank, chunk_id in enumerate(model_answer.used_chunk_ids, start=1):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunks_by_id[chunk_id]
        citations.append(
            AnswerCitation(
                chunk_id=chunk.chunk_id,
                rank=rank,
                relevance_score=chunk.retrieval_score,
                quote=chunk.full_text,
                act_title=chunk.act_title,
                doc_number=chunk.doc_number,
                doc_date=chunk.doc_date,
                structure_ref=chunk.structure_ref,
                article_no=chunk.article_no,
                clause_range=chunk.clause_range,
            )
        )
    return citations


def generate_answer(
    question: str,
    db_url: str,
    config: AppConfig,
    provider: LLMProvider,
    device: str = "auto",
    act_filter: str | None = None,
    article_no: str | None = None,
) -> GeneratedAnswer:
    """Выполняет полный generation pipeline."""
    retrieved_chunks = retrieve_chunks(
        db_url=db_url,
        question=question,
        retrieval_config=config.retrieval,
        embedding_config=config.embedding,
        device=device,
        act_filter=act_filter,
        article_no=article_no,
    )
    budget = apply_token_budget(question, retrieved_chunks, config, provider)
    messages = build_answer_messages(question, budget.selected_chunks, config.llm.prompt_version)

    started_at = time.perf_counter()
    response: LLMResponse = provider.complete(
        messages=messages,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_output_tokens,
    )
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    model_answer = parse_model_answer(response.content)
    citations = build_answer_citations(model_answer, budget.selected_chunks)

    return GeneratedAnswer(
        answer=model_answer.answer,
        used_chunk_ids=model_answer.used_chunk_ids,
        needs_clarification=model_answer.needs_clarification,
        answer_citations=citations,
        retrieval_method=config.retrieval.method,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
        dropped_chunk_ids=budget.dropped_chunk_ids,
        llm_model=response.model or config.llm.model,
        prompt_version=config.llm.prompt_version,
        latency_ms=latency_ms,
    )


def generated_answer_to_json(result: GeneratedAnswer) -> str:
    """Сериализует GeneratedAnswer в pretty JSON."""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def _strip_json_fence(content: str) -> str:
    """Удаляет markdown fence вокруг JSON, если модель его добавила."""
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
