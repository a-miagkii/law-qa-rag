from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from law_qa_rag.config import DEFAULT_SETTINGS_PATH, load_config
from law_qa_rag.env import get_database_url, load_project_dotenv
from law_qa_rag.generation import GeneratedAnswer, generate_answer
from law_qa_rag.llm.gigachat_client import GigaChatProvider, validate_model_available
from law_qa_rag.persistence import (
    authenticate_user,
    create_user,
    get_feedback_for_answer_and_user,
    load_answer_page,
    load_source_page,
    save_answer_run,
    save_feedback,
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
load_project_dotenv()
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "dev-session-secret-change-me"),
    same_site="lax",
    https_only=False,
)
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
        current_user = get_current_user(request)
        answer_id = await run_in_threadpool(
            save_answer_run,
            db_url,
            question,
            result,
            user_id=current_user["id"] if current_user else None,
        )
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
    current_user = get_current_user(request)
    try:
        db_url = _get_db_url()
        page = await run_in_threadpool(load_answer_page, db_url, answer_id)
        feedback = (
            await run_in_threadpool(
                get_feedback_for_answer_and_user,
                db_url,
                answer_id,
                current_user["id"],
            )
            if current_user
            else None
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request,
        "answer.html",
        _template_context(
            request,
            page=page,
            feedback=feedback,
            feedback_saved=request.query_params.get("feedback_saved") == "1",
        ),
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


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    """Показывает форму регистрации."""
    return templates.TemplateResponse(
        request,
        "register.html",
        _template_context(request),
    )


@app.post("/register", response_class=HTMLResponse, response_model=None)
async def register(request: Request) -> HTMLResponse | RedirectResponse:
    """Создает локального пользователя и открывает session."""
    form = await request.form()
    external_uid = _coerce_question(form.get("external_uid"))
    display_name = _coerce_question(form.get("display_name")) or None
    password = str(form.get("password") or "")

    try:
        user = await run_in_threadpool(
            create_user,
            _get_db_url(),
            external_uid,
            password,
            display_name,
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "register.html",
            _template_context(
                request,
                error=str(exc),
                external_uid=external_uid,
                display_name=display_name,
            ),
            status_code=400,
        )

    _set_session_user(request, user)
    return RedirectResponse(url=str(request.url_for("index")), status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Показывает форму входа."""
    return templates.TemplateResponse(
        request,
        "login.html",
        _template_context(request, next_url=_safe_next(request.query_params.get("next"))),
    )


@app.post("/login", response_class=HTMLResponse, response_model=None)
async def login(request: Request) -> HTMLResponse | RedirectResponse:
    """Проверяет логин и пароль."""
    form = await request.form()
    external_uid = _coerce_question(form.get("external_uid"))
    password = str(form.get("password") or "")
    next_url = _safe_next(str(form.get("next") or "/"))
    user = await run_in_threadpool(
        authenticate_user,
        _get_db_url(),
        external_uid,
        password,
    )
    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            _template_context(
                request,
                error="Неверный логин или пароль.",
                external_uid=external_uid,
                next_url=next_url,
            ),
            status_code=400,
        )

    _set_session_user(request, user)
    return RedirectResponse(url=next_url, status_code=303)


@app.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    """Завершает пользовательскую session."""
    request.session.clear()
    return RedirectResponse(url=str(request.url_for("index")), status_code=303)


@app.post("/answers/{answer_id}/feedback", response_model=None)
async def answer_feedback(request: Request, answer_id: int) -> RedirectResponse | HTMLResponse:
    """Сохраняет оценку ответа от вошедшего пользователя."""
    current_user = get_current_user(request)
    if current_user is None:
        return RedirectResponse(
            url=f"{request.url_for('login_page')}?next={request.url_for('answer_page', answer_id=answer_id).path}",
            status_code=303,
        )

    form = await request.form()
    try:
        rating = int(str(form.get("rating") or ""))
    except ValueError:
        rating = 0
    comment = str(form.get("comment") or "")

    try:
        await run_in_threadpool(
            save_feedback,
            _get_db_url(),
            answer_id,
            current_user["id"],
            rating,
            comment,
        )
    except Exception as exc:
        page = await run_in_threadpool(load_answer_page, _get_db_url(), answer_id)
        return templates.TemplateResponse(
            request,
            "answer.html",
            _template_context(
                request,
                page=page,
                feedback={"rating": rating, "comment": comment},
                feedback_error=str(exc),
            ),
            status_code=400,
        )

    return RedirectResponse(
        url=f"{request.url_for('answer_page', answer_id=answer_id)}?feedback_saved=1",
        status_code=303,
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


def get_current_user(request: Request) -> dict[str, Any] | None:
    """Возвращает текущего пользователя из session."""
    user = request.session.get("user")
    return dict(user) if isinstance(user, dict) and user.get("id") else None


def require_current_user(request: Request) -> dict[str, Any]:
    """Возвращает текущего пользователя или требует вход."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Нужен вход в систему.")
    return user


def _set_session_user(request: Request, user: dict[str, Any]) -> None:
    """Сохраняет минимальный профиль пользователя в session."""
    request.session["user"] = {
        "id": int(user["id"]),
        "external_uid": str(user["external_uid"]),
        "display_name": user.get("display_name"),
    }


def _safe_next(value: str | None) -> str:
    """Ограничивает redirect target локальными путями."""
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


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
        "current_user": get_current_user(request),
    }
    context.update(extra)
    return context
