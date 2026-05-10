from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, load_config
from law_qa_rag.env import get_database_url, load_project_dotenv
from law_qa_rag.generation import GeneratedAnswer, generate_answer
from law_qa_rag.llm.gigachat_client import GigaChatProvider, validate_model_available
from law_qa_rag.persistence import (
    load_answer_page,
    load_source_page,
    save_answer_run,
)


WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

EXAMPLE_QUESTIONS = [
    "Что такое водные объекты общего пользования?",
    "Какие обязанности есть у работника?",
    "Какие сведения относятся к персональным данным?",
]

DISCLAIMER = (
    "Ответ строится только по редакциям документов, загруженным в локальный корпус. "
    "Перед практическим применением проверьте актуальную редакцию по официальному источнику."
)


app = FastAPI(title="law-qa-rag web prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Показывает главную страницу с формой вопроса."""
    return templates.TemplateResponse(
        request,
        "index.html",
        _template_context(request),
    )


@app.post("/ask", response_model=None)
async def ask(request: Request) -> HTMLResponse | JSONResponse | RedirectResponse:
    """Принимает вопрос, генерирует и сохраняет ответ."""
    question, wants_json = await _extract_question(request)
    if not question:
        if wants_json:
            raise HTTPException(status_code=400, detail="Введите вопрос.")
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(request, error="Введите вопрос."),
            status_code=400,
        )

    try:
        db_url = _get_db_url()
        result = await run_in_threadpool(_generate_answer_for_web, question, db_url)
        answer_id = await run_in_threadpool(save_answer_run, db_url, question, result)
    except Exception as exc:
        if wants_json:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(
                request,
                error=f"Не удалось сформировать ответ: {exc}",
                question=question,
            ),
            status_code=500,
        )

    if wants_json:
        payload = result.to_dict()
        payload["answer_id"] = answer_id
        payload["answer_url"] = str(request.url_for("answer_page", answer_id=answer_id))
        return JSONResponse(payload)

    return RedirectResponse(
        url=str(request.url_for("answer_page", answer_id=answer_id)),
        status_code=303,
    )


@app.get("/answers/{answer_id}", response_class=HTMLResponse, name="answer_page")
async def answer_page(request: Request, answer_id: int) -> HTMLResponse:
    """Показывает сохраненный ответ и цитаты."""
    try:
        page = await run_in_threadpool(load_answer_page, _get_db_url(), answer_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request,
        "answer.html",
        _template_context(request, page=page),
    )


@app.get("/sources/{act_id}", response_class=HTMLResponse, name="source_page")
async def source_page(
    request: Request,
    act_id: int,
    answer_id: int | None = None,
) -> HTMLResponse:
    """Показывает акт и chunks с подсветкой процитированных chunks."""
    try:
        page = await run_in_threadpool(load_source_page, _get_db_url(), act_id, answer_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request,
        "source.html",
        _template_context(request, page=page),
    )


async def _extract_question(request: Request) -> tuple[str, bool]:
    """Достает вопрос из JSON или HTML form."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        question = _coerce_question(payload.get("question"))
        return question, True

    form = await request.form()
    question = _coerce_question(form.get("question"))
    return question, False


def _coerce_question(value: Any) -> str:
    """Преобразует входное значение вопроса в строку."""
    if value is None:
        return ""
    return str(value).strip()


def _get_db_url() -> str:
    """Возвращает DATABASE_URL или падает с понятной ошибкой."""
    db_url = get_database_url(required=True)
    if not db_url:
        raise RuntimeError("Нужен DATABASE_URL для подключения к PostgreSQL.")
    return db_url


def _generate_answer_for_web(question: str, db_url: str) -> GeneratedAnswer:
    """Запускает generation pipeline для web-запроса."""
    load_project_dotenv()
    settings_path = Path(os.getenv("SETTINGS_PATH", str(DEFAULT_SETTINGS_PATH)))
    device = os.getenv("RAG_DEVICE", "auto")
    config = load_config(settings_path)
    provider = GigaChatProvider(model=config.llm.model)
    validate_model_available(provider, config.llm.model)
    return generate_answer(
        question=question,
        db_url=db_url,
        config=config,
        provider=provider,
        device=device,
    )


def _template_context(request: Request, **extra: Any) -> dict[str, Any]:
    """Собирает общий контекст Jinja-шаблонов."""
    context = {
        "request": request,
        "examples": EXAMPLE_QUESTIONS,
        "disclaimer": DISCLAIMER,
    }
    context.update(extra)
    return context
