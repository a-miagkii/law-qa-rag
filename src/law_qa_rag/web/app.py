from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

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
    get_user_question_history,
    get_user_by_id,
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
    current_user = get_current_user(request)
    if current_user is None:
        if wants_json:
            raise HTTPException(status_code=401, detail="Чтобы задать вопрос, войдите в систему.")
        return _redirect_to_auth(request, next_url="/", login_required=True)

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
        answer_id = await run_in_threadpool(
            save_answer_run,
            db_url,
            question,
            result,
            user_id=current_user["id"],
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


@app.get(
    "/answers/{answer_id}", response_class=HTMLResponse, response_model=None, name="answer_page"
)
async def answer_page(request: Request, answer_id: int) -> HTMLResponse | RedirectResponse:
    """Показывает сохраненный ответ и цитаты."""
    current_user = get_current_user(request)
    if current_user is None:
        return _redirect_to_auth(
            request,
            next_url=request.url.path,
            login_required=True,
        )

    try:
        db_url = _get_db_url()
        page = await run_in_threadpool(load_answer_page, db_url, answer_id)
        _ensure_answer_owner(page, current_user)
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


@app.get("/sources/{act_id}", response_class=HTMLResponse, response_model=None, name="source_page")
async def source_page(
    request: Request,
    act_id: int,
    answer_id: int | None = None,
) -> HTMLResponse | RedirectResponse:
    """Показывает акт и chunks с подсветкой процитированных chunks."""
    if answer_id is not None:
        current_user = get_current_user(request)
        if current_user is None:
            return _redirect_to_auth(
                request,
                next_url=_current_path_with_query(request),
                login_required=True,
            )
        try:
            answer_page_data = await run_in_threadpool(load_answer_page, _get_db_url(), answer_id)
            _ensure_answer_owner(answer_page_data, current_user)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        page = await run_in_threadpool(load_source_page, _get_db_url(), act_id, answer_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request,
        "source.html",
        _template_context(request, page=page),
    )


@app.get("/profile", response_class=HTMLResponse, response_model=None, name="profile_page")
async def profile_page(request: Request) -> HTMLResponse | RedirectResponse:
    """Показывает профиль и историю вопросов текущего пользователя."""
    current_user = get_current_user(request)
    if current_user is None:
        return _redirect_to_auth(
            request,
            next_url=request.url.path,
            login_required=True,
        )

    history = await run_in_threadpool(
        get_user_question_history,
        _get_db_url(),
        current_user["id"],
    )
    return templates.TemplateResponse(
        request,
        "profile.html",
        _template_context(request, history=history),
    )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> RedirectResponse:
    """Открывает главную страницу с вкладкой регистрации."""
    return _redirect_to_auth(request, auth_mode="register")


@app.post("/auth/register", response_class=HTMLResponse, response_model=None, name="auth_register")
async def auth_register(request: Request) -> HTMLResponse | RedirectResponse:
    """Создает локального пользователя и открывает session."""
    form = await request.form()
    external_uid = _coerce_question(form.get("external_uid"))
    display_name = _coerce_question(form.get("display_name")) or None
    password = str(form.get("password") or "")
    password_confirm = str(form.get("password_confirm") or "")
    next_url = _safe_next(str(form.get("next") or "/"))

    try:
        if password != password_confirm:
            raise ValueError("Пароли не совпадают.")
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
            "index.html",
            _template_context(
                request,
                auth_error=str(exc),
                auth_external_uid=external_uid,
                auth_display_name=display_name,
                auth_mode="register",
                auth_modal_open=True,
                auth_next=next_url,
            ),
            status_code=400,
        )

    _set_session_user(request, user)
    return RedirectResponse(url=next_url, status_code=303)


@app.post("/register", response_class=HTMLResponse, response_model=None)
async def register(request: Request) -> HTMLResponse | RedirectResponse:
    """Совместимый alias для старого POST /register."""
    return await auth_register(request)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> RedirectResponse:
    """Открывает главную страницу с вкладкой входа."""
    return _redirect_to_auth(
        request,
        next_url=_safe_next(request.query_params.get("next")),
        auth_mode="login",
    )


@app.get("/auth", response_class=HTMLResponse)
async def auth_page(request: Request) -> HTMLResponse:
    """Fallback-страница авторизации без отдельного UI."""
    return templates.TemplateResponse(
        request,
        "index.html",
        _template_context(
            request,
            auth_modal_open=True,
            auth_mode=_auth_mode(request.query_params.get("mode")),
            auth_next=_safe_next(request.query_params.get("next")),
        ),
    )


@app.post("/auth/login", response_class=HTMLResponse, response_model=None, name="auth_login")
async def auth_login(request: Request) -> HTMLResponse | RedirectResponse:
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
            "index.html",
            _template_context(
                request,
                auth_error="Неверный логин или пароль.",
                auth_external_uid=external_uid,
                auth_mode="login",
                auth_modal_open=True,
                auth_next=next_url,
            ),
            status_code=400,
        )

    _set_session_user(request, user)
    return RedirectResponse(url=next_url, status_code=303)


@app.post("/login", response_class=HTMLResponse, response_model=None)
async def login(request: Request) -> HTMLResponse | RedirectResponse:
    """Совместимый alias для старого POST /login."""
    return await auth_login(request)


@app.post("/auth/logout", name="auth_logout")
async def auth_logout(request: Request) -> RedirectResponse:
    """Завершает пользовательскую session."""
    request.session.clear()
    return RedirectResponse(url=str(request.url_for("index")), status_code=303)


@app.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    """Совместимый alias для старого POST /logout."""
    return await auth_logout(request)


@app.post("/answers/{answer_id}/feedback", response_model=None)
async def answer_feedback(request: Request, answer_id: int) -> RedirectResponse | HTMLResponse:
    """Сохраняет оценку ответа от вошедшего пользователя."""
    current_user = get_current_user(request)
    if current_user is None:
        return _redirect_to_auth(
            request,
            next_url=request.url_for("answer_page", answer_id=answer_id).path,
            login_required=True,
        )

    form = await request.form()
    try:
        rating = int(str(form.get("rating") or ""))
    except ValueError:
        rating = 0
    comment = str(form.get("comment") or "")

    try:
        page = await run_in_threadpool(load_answer_page, _get_db_url(), answer_id)
        _ensure_answer_owner(page, current_user)
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
        _ensure_answer_owner(page, current_user)
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
    if hasattr(request.state, "current_user"):
        return request.state.current_user

    session_user_id = request.session.get("user_id")
    if session_user_id is None:
        legacy_user = request.session.get("user")
        if isinstance(legacy_user, dict):
            session_user_id = legacy_user.get("id")
    if session_user_id is None:
        request.state.current_user = None
        return None

    try:
        user = get_user_by_id(_get_db_url(), int(session_user_id))
    except Exception:
        request.session.clear()
        request.state.current_user = None
        return None
    if user is None:
        request.session.clear()
        request.state.current_user = None
        return None

    request.state.current_user = {
        "id": int(user["id"]),
        "external_uid": str(user["external_uid"]),
        "display_name": user.get("display_name"),
    }
    return request.state.current_user


def require_current_user(request: Request) -> dict[str, Any]:
    """Возвращает текущего пользователя или требует вход."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Нужен вход в систему.")
    return user


def _set_session_user(request: Request, user: dict[str, Any]) -> None:
    """Сохраняет минимальный профиль пользователя в session."""
    session_user = {
        "id": int(user["id"]),
        "external_uid": str(user["external_uid"]),
        "display_name": user.get("display_name"),
    }
    request.session["user_id"] = session_user["id"]
    request.session["user"] = session_user
    request.state.current_user = session_user


def _safe_next(value: str | None) -> str:
    """Ограничивает redirect target локальными путями."""
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def _auth_mode(value: str | None) -> str:
    """Возвращает безопасное имя вкладки авторизации."""
    return "register" if value == "register" else "login"


def _current_path_with_query(request: Request) -> str:
    """Возвращает текущий локальный путь вместе с query string."""
    path = request.url.path
    if request.url.query:
        return f"{path}?{request.url.query}"
    return path


def _redirect_to_auth(
    request: Request,
    next_url: str | None = None,
    auth_mode: str = "login",
    login_required: bool = False,
) -> RedirectResponse:
    """Перенаправляет на главную с открытым окном авторизации."""
    params = {"auth": _auth_mode(auth_mode)}
    if login_required:
        params["login_required"] = "1"
    safe_next = _safe_next(next_url)
    if safe_next:
        params["next"] = safe_next
    return RedirectResponse(
        url=f"{request.url_for('index')}?{urlencode(params)}",
        status_code=303,
    )


def _ensure_answer_owner(page: dict[str, Any], user: dict[str, Any]) -> None:
    """Проверяет, что ответ принадлежит текущему пользователю."""
    answer_user_id = page.get("answer", {}).get("user_id")
    if answer_user_id is None or int(answer_user_id) != int(user["id"]):
        raise HTTPException(status_code=403, detail="Нет доступа к этому ответу.")


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
    login_required = (
        bool(extra.pop("login_required", False))
        or request.query_params.get("login_required") == "1"
    )
    auth_mode = _auth_mode(extra.pop("auth_mode", request.query_params.get("auth")))
    auth_error = extra.pop("auth_error", None)
    auth_modal_open = (
        bool(extra.pop("auth_modal_open", False))
        or login_required
        or request.query_params.get("auth") in {"login", "register"}
        or bool(auth_error)
    )
    auth_next = _safe_next(extra.pop("auth_next", None) or request.query_params.get("next") or "/")
    context = {
        "request": request,
        "examples": EXAMPLE_QUESTIONS,
        "disclaimer": DISCLAIMER,
        "current_user": get_current_user(request),
        "login_required": login_required,
        "auth_modal_open": auth_modal_open,
        "auth_mode": auth_mode,
        "auth_error": auth_error,
        "auth_next": auth_next,
    }
    context.update(extra)
    return context
