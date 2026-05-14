# Веб-прототип: `law_qa_rag.web.app`

Веб-прототип — server-rendered интерфейс поверх текущего RAG pipeline. Он использует retrieval и LLM-настройки из `settings.yaml`, GigaChat provider из `src/law_qa_rag/llm` и сохраняет успешные ответы в PostgreSQL.

## Запуск

Сначала должен быть подготовлен корпус:

1. parsed JSON и chunks из `docs/1_parsing.md` и `docs/2_chunking.md`;
2. загруженные `acts` и `chunks` из `docs/3_load_corpus.md`;
3. рассчитанные embeddings из `docs/4_embed_chunks.md`, если retrieval использует `dense` или `weighted_hybrid`.

Локальные переменные лежат в `.env`:

```bash
cp .env.example .env
# Заполните POSTGRES_PASSWORD, SESSION_SECRET и GIGACHAT_CREDENTIALS.
```

Web-приложение автоматически читает `.env`. Если `DATABASE_URL` не задан, он собирается из `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` и `POSTGRES_PASSWORD`.
При SSL timeout или self-signed certificate для локальной разработки можно поставить `GIGACHAT_VERIFY_SSL_CERTS=false` и увеличить `GIGACHAT_TIMEOUT`.

Для существующей БД, созданной до web-этапа, один раз примените миграции:

```bash
.venv/bin/python scripts/db/apply_migration.py migrations/001_answers_web_fields.sql
.venv/bin/python scripts/db/apply_migration.py migrations/002_auth_feedback.sql
```

Первая миграция добавляет к таблице `answers` web/retrieval-поля:

- `needs_clarification`;
- `retrieval_method`;
- `retrieved_chunk_ids`;
- `dropped_chunk_ids`.

Вторая миграция добавляет к `users` поля локальной авторизации и гарантирует наличие таблицы `feedback`.

Запуск:

```bash
.venv/bin/python -m uvicorn law_qa_rag.web.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Откройте `http://127.0.0.1:8000/`.

## Endpoint `/ask`

`POST /ask` принимает вопрос двумя способами.

HTML form:

```bash
curl -i \
  -X POST \
  -F "question=Что такое водные объекты общего пользования?" \
  http://127.0.0.1:8000/ask
```

Для формы успешный ответ — `303 See Other` на `/answers/{answer_id}`.

JSON:

```bash
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question":"Что такое водные объекты общего пользования?"}' \
  http://127.0.0.1:8000/ask
```

Для JSON endpoint возвращает `answer_id`, `answer`, `used_chunk_ids`, `answer_citations`, retrieval metadata и `latency_ms`.

Если пользователь вошел в систему, `queries.user_id` будет привязан к его `users.id`.
Если пользователь не вошел, вопрос сохраняется на технического пользователя `external_uid=local-web`.

## Регистрация и вход

В web-прототипе есть минимальная локальная авторизация:

- `GET /register` — форма регистрации;
- `POST /register` — создание пользователя;
- `GET /login` — форма входа;
- `POST /login` — проверка логина и пароля;
- `POST /logout` — выход.

Логин хранится в `users.external_uid`. Это может быть email или простой username.
Пароль хранится не в открытом виде, а как PBKDF2-SHA256 hash в `users.password_hash`.

Session хранится через `SessionMiddleware`. Секрет берется из `SESSION_SECRET`; если переменная не задана, используется dev-значение только для локального запуска.

Это не промышленная система управления доступом: нет ролей администратора, восстановления пароля, подтверждения почты, внешних OAuth-провайдеров и разграничения прав доступа к корпусу.

## Обратная связь

На странице ответа есть блок «Оценка ответа».

Если пользователь вошел:

- можно поставить оценку от 1 до 5;
- можно добавить комментарий;
- повторная отправка обновляет существующую запись `feedback` по паре `(answer_id, user_id)`;
- после сохранения пользователь возвращается на ту же страницу с сообщением «Спасибо, оценка сохранена».

Если пользователь не вошел, форма скрыта и показывается ссылка на `/login`.

`feedback` используется как источник пользовательской оценки качества ответа при опытной эксплуатации прототипа.

## Страницы

- `/` — поле вопроса, кнопка отправки и примеры вопросов;
- `/answers/{answer_id}` — вопрос, ответ, предупреждение при нехватке контекста и список цитат справа;
- `/sources/{act_id}?answer_id=...` — реквизиты акта и chunks этого акта, процитированные chunks подсвечены.

На страницах выводится дисклеймер: ответ строится только по редакциям документов, загруженным в локальный корпус, и актуальность нужно проверять по официальному источнику.

## Сохранение в БД

Веб-прототип использует технического пользователя `external_uid=local-web` только для анонимных вопросов.

Перед сохранением приложение не меняет схему БД. Для существующей БД schema update выполняется явно через миграции из `migrations/`.

Успешный запуск сохраняет:

- `queries.question` и `queries.normalized_question`;
- `queries.user_id`: зарегистрированный пользователь или `local-web`;
- `answers.answer_text`, `llm_model`, `prompt_version`, `latency_ms` и retrieval metadata;
- `answer_citations.chunk_id`, `rank`, `relevance_score`, `quote`.
- `feedback.rating` и `feedback.comment`, если вошедший пользователь оценил ответ.

Ошибки GigaChat или БД показываются пользователю, но в v1 отдельно не логируются.
